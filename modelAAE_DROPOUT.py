import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastai.vision.all import *
from fastai.callback.hook import *
from pytorch_msssim import ms_ssim
from fastai.torch_core import TensorBase

# Détection de l'accélérateur matériel
if torch.cuda.is_available():
    dev = torch.device("cuda")
elif torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")

# ==============================================================================
# 1. UTILITAIRES ET COMPOSANTS U-NET (Proposition du promoteur)
# ==============================================================================

def _get_sz_change_idxs(sizes):
    """Identifie les indices où la taille spatiale des features change dans l'encodeur."""
    feature_szs = []
    for size in sizes:
        # on parcourt chaque tuple de taille (C, H, W) et on conserve la dimension spatiale
        W = size[-1]
        feature_szs.append(W)
    return list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])

class UnetBlock(Module):
    @delegates(ConvLayer.__init__)
    def __init__(self, up_in_c, x_in_c, hook,
                 final_div=True, blur=False,
                 act_cls=defaults.activation,
                 self_attention=False,
                 init=nn.init.kaiming_normal_,
                 norm_type=None,
                 skip_dropout=0.0,
                 **kwargs):

        self.hook = hook

        # Upsampling
        self.shuf = PixelShuffle_ICNR(
            up_in_c, up_in_c//2,
            blur=blur,
            act_cls=act_cls,
            norm_type=norm_type
        )

        self.bn = BatchNorm(x_in_c)

        # Dropout spatial 2D appliqué uniquement sur la skip connection
        self.skip_dropout = nn.Dropout2d(skip_dropout) if skip_dropout > 0 else None

        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2

        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(
            nf, nf,
            act_cls=act_cls,
            norm_type=norm_type,
            xtra=SelfAttention(nf) if self_attention else None,
            **kwargs
        )

        self.relu = act_cls()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    def forward(self, up_in):
        s = self.hook.stored  # Récupération des features de l'encodeur
        up_out = self.shuf(up_in)

        # Alignement des dimensions spatiales si nécessaire
        if s.shape[-2:] != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')

        s = self.bn(s)

        # Application du dropout sur les canaux spatiaux de la skip connection
        if self.skip_dropout is not None:
            s = self.skip_dropout(s)

        cat_x = self.relu(torch.cat([up_out, s], dim=1))
        return self.conv2(self.conv1(cat_x))

class ResizeToOrig(Module):
    """Redimensionne le tenseur à la taille de l'image originale stockée dans l'attribut .orig"""
    def __init__(self, mode='nearest'):
        self.mode = mode

    def forward(self, x):
        if x.orig.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, x.orig.shape[-2:], mode=self.mode)
        return x

class DynamicUnetSkipDropout(SequentialEx):
    """U-Net dynamique intégrant le Dropout2d sur les skip connections."""
    def __init__(self, encoder, n_out, img_size,
                 blur=False, blur_final=True,
                 self_attention=False,
                 y_range=None,
                 last_cross=True,
                 bottle=False,
                 act_cls=defaults.activation,
                 init=nn.init.kaiming_normal_,
                 norm_type=None,
                 skip_dropout=0.0,
                 **kwargs):

        imsize = img_size
        sizes = model_sizes(encoder, size=imsize)
        sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))

        # Initialisation des hooks sur l'encodeur
        self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)

        x = dummy_eval(encoder, imsize).detach()
        ni = sizes[-1][1]

        # Bottleneck standard du U-Net
        middle_conv = nn.Sequential(
            ConvLayer(ni, ni*2, act_cls=act_cls, norm_type=norm_type, **kwargs),
            ConvLayer(ni*2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs)
        ).eval()

        x = middle_conv(x)
        layers = [encoder, BatchNorm(ni), nn.ReLU(), middle_conv]

        # Construction du décodeur
        for i, idx in enumerate(sz_chg_idxs):
            not_final = i != len(sz_chg_idxs)-1

            up_in_c = int(x.shape[1])
            x_in_c  = int(sizes[idx][1])

            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sz_chg_idxs)-3)

            unet_block = UnetBlock(
                up_in_c, x_in_c, self.sfs[i],
                final_div=not_final,
                blur=do_blur,
                self_attention=sa,
                act_cls=act_cls,
                init=init,
                norm_type=norm_type,
                skip_dropout=skip_dropout,
                **kwargs
            ).eval()

            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]

        if imsize != sizes[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))

        layers.append(ResizeToOrig())

        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(
                ResBlock(
                    1, ni, ni//2 if bottle else ni,
                    act_cls=act_cls, norm_type=norm_type, **kwargs
                )
            )

        layers += [ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=norm_type, **kwargs)]
        apply_init(nn.Sequential(layers[3], layers[-2]), init)

        if y_range is not None:
            layers.append(SigmoidRange(*y_range))

        layers.append(ToTensorBase())
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()

# ==============================================================================
# 2. ARCHITECTURE AAE PRINCIPALE
# ==============================================================================
class AAE(nn.Module):
    def __init__(
        self,
        input_size=256,
        input_channels=3,
        encoding_dims=128,
        classes=2, 
        gen_train=True,
        skip_dropout=1 # Remplacement du skip_weight par skip_dropout 
    ):
        super(AAE, self).__init__()

        self.gen_train = gen_train
        self.count_acc = 1
        self.classes = classes
        
        # A. Encodeur de base
        encoder_base = nn.Sequential(*list(resnet34(weights=ResNet34_Weights.DEFAULT).children())[:-2])
        
        # Instanciation de la nouvelle architecture U-Net
        self.unet = DynamicUnetSkipDropout(
            encoder=encoder_base, 
            n_out=input_channels, 
            img_size=(input_size, input_size), 
            skip_dropout=skip_dropout,
            last_cross=False, # Crucial : désactive la connexion résiduelle de l'image source à la sortie
            y_range=(0.0, 1.0) #ajout
        )
        
        # B. Bottleneck xAI
        flat_features = 512 * 8 * 8
        self.flatten = nn.Flatten()
        
        self.fc_encode = nn.Linear(flat_features, encoding_dims)
        self.bn_lin = nn.BatchNorm1d(num_features=encoding_dims)
        self.decoder_fc = nn.Linear(encoding_dims, flat_features)

        # C. Têtes du réseau
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(encoding_dims, self.classes, bias=True) 

        self.fc_crit1 = nn.Linear(encoding_dims, 64)
        self.fc_crit2 = nn.Linear(64, 16)
        self.fc_crit3 = nn.Linear(16, 1)

        self.bn_crit1 = nn.BatchNorm1d(num_features=64)
        self.bn_crit2 = nn.BatchNorm1d(num_features=16)

    def latent_gan(self, z: Tensor) -> Tensor:

        # on test en retirant le batch norm :self.bn_crit1 et self.bn_crit2
        x = F.leaky_relu((self.fc_crit1(z)), negative_slope=0.2)
        x = F.leaky_relu((self.fc_crit2(x)),  negative_slope=0.2)
        x = torch.sigmoid(self.fc_crit3(x)) 
        return x
    
    def denoising_ae_loss_func(self, clean_xb,RECONS_WEIGHT,CLASS_WEIGHT, pred, yb):
        # pred et yb sont ignorés car cette fonction est dédiée au pré-entraînement de l'AE en mode débruitage
        # Fastai attend une signature de fonction de perte avec ces arguments, même si on ne les utilise pas tous
        #Partie loss recon
        alpha = 0.84
        l1_loss = F.l1_loss(self.decoder_output, clean_xb)
        ms_ssim_val = ms_ssim(self.decoder_output, clean_xb, data_range=1.0, size_average=True)
        msssim_loss = 1.0 - ms_ssim_val
        self.recons_loss = alpha * msssim_loss + (1.0 - alpha) * l1_loss
        #partie loss adversarial
        adversarial_loss = nn.BCELoss()
        if self.gen_train: 
            valid = torch.ones_like(self.gan_fake, requires_grad=False).detach()
            self.adv_loss = adversarial_loss(self.gan_fake, valid)
            self.crit_loss = 0
        else:
            valid = torch.ones_like(self.gan_real, requires_grad=False).detach()
            fake = torch.zeros_like(self.gan_fake, requires_grad=False).detach()
            self.real_loss = adversarial_loss(self.gan_real, valid)
            self.fake_loss = adversarial_loss(self.gan_fake, fake)
            self.adv_loss = 0.6 * self.real_loss + 0.4 * self.fake_loss
            self.crit_loss = self.adv_loss


        total_loss = RECONS_WEIGHT* self.recons_loss +CLASS_WEIGHT*self.adv_loss
        
        return total_loss

    def classif_loss_func(self, output, target,ADV_WEIGHT, RECONS_WEIGHT, CLASS_WEIGHT, **kwargs):
        alpha = 0.84
        # On utilise self.input_image car il n'y a pas de corruption ici
        l1_loss = F.l1_loss(self.decoder_output, self.input_image)
        ms_ssim_val = ms_ssim(self.decoder_output, self.input_image, data_range=1.0, size_average=True)
        msssim_loss = 1.0 - ms_ssim_val
        self.recons_loss = alpha * msssim_loss + (1.0 - alpha) * l1_loss

        # LOn sauvegarde l'attribut pour LossAttrMetric
        self.classif_loss = F.cross_entropy(output, target, **kwargs)
        #partie loss adversarial
        adversarial_loss = nn.BCELoss()
        if self.gen_train: 
            valid = torch.ones_like(self.gan_fake, requires_grad=False).detach()
            self.adv_loss = adversarial_loss(self.gan_fake, valid)
            self.crit_loss = 0
        else:
            valid = torch.ones_like(self.gan_real, requires_grad=False).detach()
            fake = torch.zeros_like(self.gan_fake, requires_grad=False).detach()
            self.real_loss = adversarial_loss(self.gan_real, valid)
            self.fake_loss = adversarial_loss(self.gan_fake, fake)
            self.adv_loss = 0.6 * self.real_loss + 0.4 * self.fake_loss
            self.crit_loss = self.adv_loss







        loss = ADV_WEIGHT * self.adv_loss + RECONS_WEIGHT * self.recons_loss + CLASS_WEIGHT * self.classif_loss
        return loss
    
    def pure_classif_loss_func(self,pred, target, **kwargs):

        return F.cross_entropy(pred, target, **kwargs)
    
    def aae_loss_func(self, output, target):
        
        
        

        adversarial_loss = nn.BCELoss()
        if self.gen_train: 
            valid = torch.ones_like(self.gan_fake, requires_grad=False).detach()
            self.adv_loss = adversarial_loss(self.gan_fake, valid)
            self.crit_loss = 0
        else:
            valid = torch.ones_like(self.gan_real, requires_grad=False).detach()
            fake = torch.zeros_like(self.gan_fake, requires_grad=False).detach()
            self.real_loss = adversarial_loss(self.gan_real, valid)
            self.fake_loss = adversarial_loss(self.gan_fake, fake)
            self.adv_loss = 0.6 * self.real_loss + 0.4 * self.fake_loss
            self.crit_loss = self.adv_loss

        

        loss = self.adv_loss
            
        return loss

    def forward(self, x):
        self.input_image = x

        # =========================================================
        # ÉTAPE 1 : ENCODEUR
        # =========================================================
        # Déclenche silencieusement la sauvegarde des skip connections (self.unet.sfs)
        feats = self.unet.layers[0](x)

        # =========================================================
        # ÉTAPE 2 : BOTTLENECK AAE
        # =========================================================
        flat = self.flatten(feats)
        self.zi = self.fc_encode(flat)
        
        labels = self.linear(self.zi)
        
        self.gan_fake = self.latent_gan(self.zi)
        z_random = torch.randn_like(self.zi)
        self.gan_real = self.latent_gan(z_random)

        # =========================================================
        # ÉTAPE 3 : DÉCODEUR
        # =========================================================
        z_spatial = F.relu(self.decoder_fc(self.zi))
        z_spatial = z_spatial.view(-1, 512, 8, 8) 
        
        out = TensorBase(z_spatial)
        
        # Maintien du tenseur fantôme pour la compatibilité avec ResizeToOrig
        ghost_shape = torch.zeros_like(self.input_image)
        orig_x = TensorBase(ghost_shape)
        
        # Parcours du décodeur (inclut le bottleneck standard Fastai et les UnetBlocks)
        for layer in self.unet.layers[1:]:
            out.orig = orig_x
            nres = layer(out)
            
            # Nettoyage VRAM
            out.orig = None
            if hasattr(nres, 'orig'):
                nres.orig = None
                
            out = nres
            
        self.decoder_output = out

        return labels

    # Les fonctions de pertes (denoising_ae_loss_func, ae_loss_func, etc.) 
    # restent strictement identiques à ton code original et ont été omises ici
    # pour des raisons de concision, tu peux les recoller directement.