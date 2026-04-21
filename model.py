import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from fastai.data.all import *
from pytorch_msssim import ms_ssim

if torch.cuda.is_available():
    # Sur le serveur (tu peux préciser "cuda:3" si tu veux un GPU spécifique)
    dev = torch.device("cuda") 
elif torch.backends.mps.is_available():
    # Sur ton Mac (Apple Silicon M1/M2/M3...)
    dev = torch.device("mps")
else:
    # Par défaut (si ni GPU Nvidia ni puce Apple)
    dev = torch.device("cpu")


class AAE(nn.Module):
    def __init__(
        self,
        input_size,
        input_channels,
        encoding_dims=128,
        step_channels=16, 
        nonlinearity=nn.LeakyReLU(0.2),
        classes=2,
        gen_train=True
    ):
        super(AAE, self).__init__()

        self.gen_train = gen_train
        self.count_acc = 1
        self.classes = classes

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)#, inplace=True)
        # self.linear = nn.Linear(self.encoder.out_channels[-1], 2, bias=True) #2 classes
        self.linear = nn.Linear(encoding_dims, self.classes, bias=True) #8 classes
        self.bn_lin = nn.BatchNorm1d(num_features=encoding_dims)

        self.fc_crit1 = nn.Linear(encoding_dims, 64)
        self.fc_crit2 = nn.Linear(64, 16)
        self.fc_crit3 = nn.Linear(16, 1)

        self.bn_crit1 = nn.BatchNorm1d(num_features=64)
        self.bn_crit2 = nn.BatchNorm1d(num_features=16)

        encoder = [
            nn.Sequential(
                nn.Conv2d(input_channels, step_channels, 5, 2, 2), nonlinearity
            )
        ]
        size = input_size // 2
        channels = step_channels
        while size > 1:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 4, 5, 4, 2),
                    nn.BatchNorm2d(channels * 4),
                    nonlinearity,
                )
            )
            channels *= 4
            size = size // 4
        self.encoder = nn.Sequential(*encoder)
        self.encoder_fc = nn.Linear(
            channels, encoding_dims
        ) 
        # Channels correspond à 4096 pour input_size=256

        self.decoder_fc = nn.Linear(encoding_dims, channels)
        # --- DECODEUR ---
        decoder_channels = [channels//4, channels//16, channels//64, channels//256] # 4 couches comme à l'encodeur 
        scale = 4

        decoder = []
        in_ch = channels # Doit correspondre à la sortie de self.decoder_fc
        
        for out_ch in decoder_channels:
            decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2), 
                nn.BatchNorm2d(out_ch),
                nonlinearity
            ))
            in_ch = out_ch
        # --- LA COUCHE FINALE DE RESTITUTION ---
        decoder.append(nn.Sequential(
            nn.Conv2d(in_ch, input_channels, kernel_size=5, spadding=2),
            nn.Sigmoid()
        ))
        
        self.decoder = nn.Sequential(*decoder)


    def latent_gan(self, zi: Tensor) -> Tensor:
         # zi shape : (batch_size, 128) — échantillons individuels
        x = F.leaky_relu(self.bn_crit1(self.fc_crit1(zi)), negative_slope=0.2)
        x = F.leaky_relu(self.bn_crit2(self.fc_crit2(x)),  negative_slope=0.2)
        x = torch.sigmoid(self.fc_crit3(x))  # (batch_size, 1)
        return x
    
    def denoising_ae_loss_func(self, clean_xb, pred, yb):
        """
        clean_xb : image originale propre (target de reconstruction) [0, 1]
        pred     : labels de classification (sortie du forward)
        yb       : labels de classification réels
        """
        # Poids de la perte hybride (démontré optimal dans la littérature)
        alpha = 0.84
        
        # 1. Perte L1 : Maintient la colorimétrie (évite les décalages spectraux de la SSIM seule)
        l1_loss = F.l1_loss(self.decoder_output, clean_xb)
        
        # 2. Perte MS-SSIM : Force la reconstruction des structures et textures
        ms_ssim_val = ms_ssim(self.decoder_output, clean_xb, data_range=1.0, size_average=True)
        msssim_loss = 1.0 - ms_ssim_val
        
        # 3. Combinaison
        self.recons_loss = alpha * msssim_loss + (1.0 - alpha) * l1_loss
        
        return self.recons_loss 

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.input_image = x

        features = self.encoder(x) 
        self.zi = F.leaky_relu(self.bn_lin(self.encoder_fc(features.view(-1, features.size(1) * features.size(2) * features.size(3)))), negative_slope=0.2)

        z = self.decoder_fc(self.zi)     
        
        z_spatial = z.view(z.size(0), -1, 1, 1) # 4096 = channels finaux de l'encodeur pour input_size=256
        
        # On envoie dans le décodeur convolutif
        self.decoder_output = self.decoder(z_spatial)

        self.gan_fake = self.latent_gan(self.zi)
        z = torch.randn_like(self.zi)
        self.gan_real = self.latent_gan(z)

        # x = self.dropout(self.zi)
        labels = self.linear(self.zi)
        # labels = F.softmax(x)

        return labels

    def ae_loss_func(self, output, target):
        delta = .5
        huber = nn.HuberLoss(delta=delta)

        self.recons_loss = huber(self.input_image, self.decoder_output)

        bce = nn.BCEWithLogitsLoss()
        classif_loss = bce(output, target)
            
        return self.recons_loss + .001*classif_loss
    
    def pure_classif_loss_func(self, output, target, **kwargs):
        """
        Fonction de perte pour classification à 2 classes.
        Accepte **kwargs pour supporter l'argument 'reduction' envoyé par Fastai 
        lors de l'interprétation.
        """
        # F.cross_entropy fait exactement la même chose que nn.CrossEntropyLoss,
        # mais sous forme de fonction, ce qui permet de lui passer facilement les kwargs.
        return F.cross_entropy(output, target, **kwargs)

    def classif_loss_func(self, output, target):
        delta = .5
        huber = nn.HuberLoss(delta=delta)

        self.recons_loss = huber(self.input_image, self.decoder_output)

        bce = nn.BCEWithLogitsLoss()
        self.classif_loss = bce(output, target)

        # self.kld_loss = -0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp())
        adversarial_loss = nn.BCELoss()
        if self.gen_train: #generator loss
            # Measures generator's ability to fool the discriminator
            valid = torch.ones_like(self.gan_fake, requires_grad=False).detach()
            self.adv_loss = adversarial_loss(self.gan_fake, valid)
            self.crit_loss = 0
        else: #discriminator loss
            # Measure discriminator's ability to classify real from generated samples
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

        # self.kld_loss = -0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp())
    
        if self.gen_train: #generator loss
            # Measures generator's ability to fool the discriminator
            valid = torch.ones_like(self.gan_fake, requires_grad=False).detach()
            self.adv_loss = adversarial_loss(self.gan_fake, valid)
            self.crit_loss = 0
            # self.classif_loss = self.classif_loss_func(self.pred, classif_target)
            # loss = 0.1 * self.adv_loss + 0.9 * self.recons_loss + self.classif_loss
        else: #discriminator loss
            # Measure discriminator's ability to classify real from generated samples
            valid = torch.ones_like(self.gan_real, requires_grad=False).detach()
            fake = torch.zeros_like(self.gan_fake, requires_grad=False).detach()
            self.real_loss = adversarial_loss(self.gan_real, valid)
            self.fake_loss = adversarial_loss(self.gan_fake, fake)
            self.adv_loss = 0.6 * self.real_loss + 0.4 * self.fake_loss
            self.crit_loss = self.adv_loss

        # ce = nn.CrossEntropyLoss()
        # self.classif_loss = ce(output, target)
        bce = nn.BCEWithLogitsLoss()
        self.classif_loss = bce(output, target)

        # loss = self.adv_loss + .1*self.recons_loss + .4*self.classif_loss
        loss = self.adv_loss

        # print(f'Losses: {loss.shape, self.kld_loss.shape, self.recons_loss.shape, self.classif_loss.shape}')
            
        # if self.count_acc % 2 == 0:
        #     self.gen_train = False
        # else:
        #     self.gen_train = True
        # self.count_acc += 1
        # # print(f'count_acc: {self.count_acc, self.gen_train}')
            
        return loss