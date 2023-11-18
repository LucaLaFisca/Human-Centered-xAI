import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fastai.vision.all import *
from fastai.data.all import *

dev = 'cuda:3'
torch.cuda.set_device(dev)

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

        self.fc_crit1 = nn.Linear(encoding_dims*2, 64)
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
        mu = torch.mean(zi,dim=0).unsqueeze(0)
        std = torch.std(zi,dim=0).unsqueeze(0)
        # print(f'std shape: {std.shape}')
        stat = torch.hstack((mu,std))
        # print(f'stat shape: {stat.shape}')
        x = self.fc_crit1(stat)
        # print(x.grad)
        # x = F.leaky_relu(self.bn_crit1(x),negative_slope=0.2)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.fc_crit2(x)
        # print(x.grad)
        # x = F.leaky_relu(self.bn_crit2(x),negative_slope=0.2)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.fc_crit3(x)
        # print(x.grad)
        x = F.sigmoid(x)
        # print(x.grad)
        return x

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.input_image = x

        features = self.encoder(x)
        self.zi = F.relu(self.bn_lin(self.encoder_fc(
                    features.view(
                        -1, features.size(1) * features.size(2) * features.size(3)
                    )
                )))

        x = self.decoder_fc(self.zi)
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

        recons_loss = huber(self.input_image, self.decoder_output)

        bce = nn.BCEWithLogitsLoss()
        classif_loss = bce(output, target)
            
        return recons_loss + .001*classif_loss

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
        loss = self.adv_loss + .1*self.recons_loss + .001*self.classif_loss

        # print(f'Losses: {loss.shape, self.kld_loss.shape, self.recons_loss.shape, self.classif_loss.shape}')
            
        if self.count_acc % 2 == 0:
            self.gen_train = False
        else:
            self.gen_train = True
        self.count_acc += 1
        # print(f'count_acc: {self.count_acc, self.gen_train}')
            
        return loss