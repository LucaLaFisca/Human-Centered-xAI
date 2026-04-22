import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from fastai.vision.models.unet import DynamicUnet, UnetBlock
from pytorch_msssim import ms_ssim
import types

#debug pour injecter l'image d'éntrée comme l'U-Net de Fastai s'y attend
from fastai.torch_core import TensorBase

if torch.cuda.is_available():
    dev = torch.device("cuda") 
elif torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")

# ==============================================================================
# 1. LA CLASSE DU U-NET PONDÉRÉ
# ==============================================================================
class WeightedDynamicUnet(DynamicUnet):
    """
    Surcharge du DynamicUnet de Fastai pour intégrer une pondération
    spécifique sur les skip connections.
    """
    def __init__(self, encoder, n_out, img_size, skip_weight=0.1, **kwargs):
        super().__init__(encoder, n_out, img_size, **kwargs)
        self.skip_weight = skip_weight
        self._apply_weight_to_blocks()

    def _apply_weight_to_blocks(self):
        """
        Intercepte la méthode forward de chaque UnetBlock pour multiplier
        le tenseur de la skip connection (hook.stored) par notre poids.
        """
        for layer in self.layers:
            if isinstance(layer, UnetBlock):
                original_forward = layer.forward

                def weighted_forward(self_block, up_in, orig_fwd=original_forward, w=self.skip_weight):
                    # On vérifie que le hook a bien capturé le tenseur de l'encodeur
                    if hasattr(self_block, 'hook') and hasattr(self_block.hook, 'stored'):
                        # C'EST ICI LA MAGIE : On atténue l'information spatiale
                        self_block.hook.stored = self_block.hook.stored * w
                    
                    return orig_fwd(up_in)
                
                # On lie la nouvelle fonction à l'instance du bloc
                layer.forward = types.MethodType(weighted_forward, layer)


# ==============================================================================
# 2. TON ARCHITECTURE AAE PRINCIPALE
# ==============================================================================
class AAE(nn.Module):
    def __init__(
        self,
        input_size=256,
        input_channels=3,
        encoding_dims=128,
        classes=2, # Fixé à 2 pour l'attribut 'Male'
        gen_train=True,
        skip_weight=0.1 # minimale pour faire transiter un minimum par la skip connection mais pas rien non plus !
    ):
        super(AAE, self).__init__()

        self.gen_train = gen_train
        self.count_acc = 1
        self.classes = classes
        
        # ---------------------------------------------------------
        # A. Création de la structure U-Net de base (ResNet34)
        # ---------------------------------------------------------
        # On utilise le resnet34 sans sa tête de classification globale
        encoder_base = nn.Sequential(*list(resnet34(weights=ResNet34_Weights.DEFAULT).children())[:-2])
        
        # On instancie notre Unet modifié.
        # Attention : Fastai attache ses hooks sur encoder_base pendant cette instanciation
        self.unet = WeightedDynamicUnet(
            encoder=encoder_base, 
            n_out=input_channels, 
            img_size=(input_size, input_size), 
            skip_weight=skip_weight,
            last_cross=False    # U-Net par définition utilise l'image d'éntrée pour reconstruire l'image de sortie le plus parfaitement possible ; ON NE VEUT PAS CA !
        )
        
        # ---------------------------------------------------------
        # B. Création de ton goulot d'étranglement (Bottleneck xAI)
        # ---------------------------------------------------------
        # Pour une image 256x256, le ResNet34 sort un tenseur de dimensions [Batch, 512, 8, 8]
        # 512 * 8 * 8 = 32768
        flat_features = 512 * 8 * 8
        
        self.flatten = nn.Flatten()
        
        # Compression vers z (l'espace latent)
        self.fc_encode = nn.Linear(flat_features, encoding_dims)
        self.bn_lin = nn.BatchNorm1d(num_features=encoding_dims)
        
        # Décompression depuis z (pour relancer le décodeur spatial)
        self.decoder_fc = nn.Linear(encoding_dims, flat_features)

        # ---------------------------------------------------------
        # C. Têtes de ton réseau (Classifieur et GAN)
        # ---------------------------------------------------------
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(encoding_dims, self.classes, bias=True) 

        self.fc_crit1 = nn.Linear(encoding_dims, 64)
        self.fc_crit2 = nn.Linear(64, 16)
        self.fc_crit3 = nn.Linear(16, 1)

        self.bn_crit1 = nn.BatchNorm1d(num_features=64)
        self.bn_crit2 = nn.BatchNorm1d(num_features=16)

    def latent_gan(self, zi: Tensor) -> Tensor:
        x = F.leaky_relu(self.bn_crit1(self.fc_crit1(zi)), negative_slope=0.2)
        x = F.leaky_relu(self.bn_crit2(self.fc_crit2(x)),  negative_slope=0.2)
        x = torch.sigmoid(self.fc_crit3(x)) 
        return x

    def forward(self, x):
        self.input_image = x

        # =========================================================
        # ÉTAPE 1 : L'ENCODEUR (Extraction et Hooks)
        # =========================================================
        # unet.layers[0] correspond à l'encodeur ResNet34. 
        # Le fait de passer x ici déclenche silencieusement la sauvegarde 
        # des skip connections dans les hooks de Fastai.
        feats = self.unet.layers[0](x) # Shape: [Batch, 512, 8, 8]

        # =========================================================
        # ÉTAPE 2 : BOTTLENECK
        # =========================================================
        flat = self.flatten(feats)
        self.zi = F.leaky_relu(self.bn_lin(self.fc_encode(flat)), negative_slope=0.2)
        
        # Classification sur l'attribut 'Male'
        labels = self.linear(self.zi)
        
        # Architecture GAN sur l'espace latent
        self.gan_fake = self.latent_gan(self.zi)
        z_random = torch.randn_like(self.zi)
        self.gan_real = self.latent_gan(z_random)

       # =========================================================
        # ÉTAPE 3 : LE DÉCODEUR (Reconstruction assistée)
        # =========================================================
        # On redéploie le vecteur latent sous forme de matrice spatiale
        z_spatial = F.relu(self.decoder_fc(self.zi))
        z_spatial = z_spatial.view(-1, 512, 8, 8) 
        
        # ASTUCE FASTAI : Convertir en TensorBase pour autoriser l'attribut caché '.orig'
        out = TensorBase(z_spatial)
        # CRÉATION DU FANTÔME : Un tenseur vide de la même taille que l'image
        # Garantie absolue qu'aucun pixel n'est transmis par cette voie
        ghost_shape = torch.zeros_like(self.input_image)
        orig_x = TensorBase(ghost_shape)
        
        # On passe ce tenseur dans le reste du U-Net en mimant le comportement de SequentialEx
        for layer in self.unet.layers[1:]:
            
            # 1. On attache l'image originale comme "sac à dos"
            out.orig = orig_x
            
            # 2. On passe dans la couche (MergeLayer et ResizeToOrig fonctionneront !)
            nres = layer(out)
            
            # 3. Nettoyage mémoire OBLIGATOIRE (pour éviter une nouvelle fuite de VRAM GPU)
            out.orig = None
            if hasattr(nres, 'orig'):
                nres.orig = None
                
            out = nres
            
        self.decoder_output = out

        return labels

    # ==========================================================================
    # FONCTIONS DE PERTES (Conservées strictement identiques à ton code source)
    # ==========================================================================
    def denoising_ae_loss_func(self, clean_xb, pred, yb):
        alpha = 0.84
        l1_loss = F.l1_loss(self.decoder_output, clean_xb)
        ms_ssim_val = ms_ssim(self.decoder_output, clean_xb, data_range=1.0, size_average=True)
        msssim_loss = 1.0 - ms_ssim_val
        self.recons_loss = alpha * msssim_loss + (1.0 - alpha) * l1_loss
        return self.recons_loss 

    def ae_loss_func(self, output, target):
        delta = .5
        huber = nn.HuberLoss(delta=delta)
        self.recons_loss = huber(self.input_image, self.decoder_output)
        bce = nn.BCEWithLogitsLoss()
        classif_loss = bce(output, target)
        return self.recons_loss + .001*classif_loss
    
    def pure_classif_loss_func(self, output, target, **kwargs):
        return F.cross_entropy(output, target, **kwargs)

    def classif_loss_func(self, output, target):
        delta = .5
        huber = nn.HuberLoss(delta=delta)
        self.recons_loss = huber(self.input_image, self.decoder_output)
        bce = nn.BCEWithLogitsLoss()
        self.classif_loss = bce(output, target)

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
            return self.adv_loss

        loss = 0.01*self.recons_loss + 0.24*self.adv_loss + 0.75*self.classif_loss

        if self.count_acc % 16 == 0:
            self.gen_train = False
        else:
            self.gen_train = True
        self.count_acc += 1
            
        return loss

    def aae_loss_func(self, output, target):
        adversarial_loss = nn.BCELoss()
        delta = .5
        huber = nn.HuberLoss(delta=delta)
        self.recons_loss = huber(self.input_image, self.decoder_output)

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

        bce = nn.BCEWithLogitsLoss()
        self.classif_loss = bce(output, target)

        loss = self.adv_loss + .1*self.recons_loss + .001*self.classif_loss
            
        if self.count_acc % 2 == 0:
            self.gen_train = False
        else:
            self.gen_train = True
        self.count_acc += 1
            
        return loss