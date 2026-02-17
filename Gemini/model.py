import torch
import torch.nn as nn
import torch.nn.functional as F

class AAE(nn.Module):
    def __init__(
        self,
        input_size=128,
        input_channels=3,
        encoding_dims=128,
        step_channels=16,
        classes=2,
        nonlinearity=None # Default moved to body
    ):
        super(AAE, self).__init__()
        
        self.encoding_dims = encoding_dims
        self.classes = classes
        nl = nonlinearity if nonlinearity is not None else nn.LeakyReLU(0.2, inplace=True)

        # --- Encoder ---
        encoder_layers = [
            nn.Sequential(
                nn.Conv2d(input_channels, step_channels, 5, 2, 2), nl
            )
        ]
        
        curr_size = input_size // 2
        curr_channels = step_channels
        
        # Downsampling blocks
        while curr_size > 1:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_channels, curr_channels * 4, 5, 4, 2),
                    nn.BatchNorm2d(curr_channels * 4),
                    nl,
                )
            )
            curr_channels *= 4
            curr_size = curr_size // 4
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent Space
        self.encoder_fc = nn.Linear(curr_channels, encoding_dims)
        self.bn_lin = nn.BatchNorm1d(num_features=encoding_dims)
        
        # --- Heads ---
        # 1. Classifier Head
        self.classifier = nn.Linear(encoding_dims, classes)
        
        # 2. Discriminator Head (Latent GAN)
        # Input is mean + std of the batch (size 2 * encoding_dims)
        self.discriminator = nn.Sequential(
            nn.Linear(encoding_dims * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # --- Decoder ---
        self.decoder_fc = nn.Linear(encoding_dims, step_channels)
        
        decoder_layers = []
        curr_size = 1
        curr_channels = step_channels
        
        # Upsampling blocks (mirroring encoder logic)
        while curr_size < input_size // 2:
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(curr_channels, curr_channels * 4, 5, 4, 2, 3),
                    nn.BatchNorm2d(curr_channels * 4),
                    nl,
                )
            )
            curr_channels *= 4
            curr_size *= 4
            
        decoder_layers.append(nn.ConvTranspose2d(curr_channels, input_channels, 5, 2, 2, 1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # 1. Encode
        features = self.encoder(x)
        # Flatten features
        flat_features = features.view(features.size(0), -1)
        
        # Get Latent Vector (z)
        zi = self.encoder_fc(flat_features)
        zi = F.relu(self.bn_lin(zi))

        # 2. Decode
        dec_input = self.decoder_fc(zi)
        dec_input = dec_input.view(-1, dec_input.size(1), 1, 1)
        reconstruction = self.decoder(dec_input)

        # 3. Classify
        logits = self.classifier(zi)

        # 4. Discriminate (Latent GAN)
        # Calculate stats for the batch (Feature Matching style AAE)
        mu = torch.mean(zi, dim=0).unsqueeze(0)
        std = torch.std(zi, dim=0).unsqueeze(0)
        
        # Handle single-item batches edge case
        if torch.isnan(std).any(): 
            std = torch.zeros_like(mu)

        real_stats = torch.cat((mu, std), dim=1)
        disc_fake = self.discriminator(real_stats)
        
        # Generate Prior (Gaussian) for Discriminator
        z_prior = torch.randn_like(zi)
        mu_prior = torch.mean(z_prior, dim=0).unsqueeze(0)
        std_prior = torch.std(z_prior, dim=0).unsqueeze(0)
        prior_stats = torch.cat((mu_prior, std_prior), dim=1)
        
        disc_real = self.discriminator(prior_stats)

        # Return dictionary for flexibility in Loss function
        return {
            'input': x,
            'reconstruction': reconstruction,
            'logits': logits,
            'zi': zi,
            'disc_fake': disc_fake, # Prob that encoded Z is real
            'disc_real': disc_real  # Prob that Gaussian Noise is real
        }