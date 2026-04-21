import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from fastai.data.all import *

# dev = 'cuda:3'
# torch.cuda.set_device(dev)
dev = torch.cuda.current_device()


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
        )  # Can add a Tanh nonlinearity if training is unstable as noise prior is Gaussian
        self.decoder_fc = nn.Linear(encoding_dims, step_channels)
        decoder = []
        size = 1
        channels = step_channels
        while size < input_size // 2:
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels * 4, 5, 4, 2, 3),
                    nn.BatchNorm2d(channels * 4),
                    nonlinearity,
                )
            )
            channels *= 4
            size *= 4
        decoder.append(nn.ConvTranspose2d(channels, input_channels, 5, 2, 2, 1))
        self.decoder = nn.Sequential(*decoder)


    def latent_gan(self, zi: Tensor) -> Tensor:
        # zi shape : (batch_size, 128) — échantillons individuels
        x = F.leaky_relu(self.bn_crit1(self.fc_crit1(zi)), negative_slope=0.2)
        x = F.leaky_relu(self.bn_crit2(self.fc_crit2(x)),  negative_slope=0.2)
        x = torch.sigmoid(self.fc_crit3(x))  # (batch_size, 1)
        return x

    def forward(self, x):
        print(f"\n--- DEBUT FORWARD ---")
        print(f"1. Image entrée 'x' : {x.shape}")
        
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.input_image = x

        features = self.encoder(x) # modifier relu en leaky_relu
        print(f"-> Après convolution (features) : {features.shape}")
        # self.zi = F.leaky_relu(self.bn_lin(self.encoder_fc(
        #             features.view(
        #                 -1, features.size(1) * features.size(2) * features.size(3)
        #             )
        #         )))
        # Test self.zi avec batch norm et sans leaky relu 
        #test sans batch norm self.bn_lin
        self.zi = (self.encoder_fc(
            features.view(
                -1, features.size(1) * features.size(2) * features.size(3)
            ))
        )
        print(f"-> Après la couche linéaire (zi) : {self.zi.shape}")
        x = self.decoder_fc(self.zi)
        print(f"3. Sortie Décodeur 'recons' : {x.shape}")
        self.decoder_output = self.decoder(x.view(-1, x.size(1), 1, 1))

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
        
        # # 1. On regarde la mémoire juste avant le calcul (en Mo)
        # mem_before = torch.cuda.memory_allocated() / 1024**2
        # print(f"-> Entrée Loss | Mem: {mem_before:.1f} Mo | Preds: {output.shape} | Targets: {target.shape}")
        
        
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
            #loss = 0.2 * self.adv_loss + 0.8 * self.recons_loss + self.classif_loss
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
        loss = self.adv_loss #+ .99*self.recons_loss changed self.recons from 0.1 to 0.05

        # print(f'Losses: {loss.shape, self.kld_loss.shape, self.recons_loss.shape, self.classif_loss.shape}')
            
        # if self.count_acc % 2 == 0:
        #     self.gen_train = False
        # else:
        #     self.gen_train = True
        # self.count_acc += 1
        # print(f'count_acc: {self.count_acc, self.gen_train}')
            
        return loss