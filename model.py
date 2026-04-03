import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def encoder_block(in_ch, out_ch):
    """Conv2d  + BN + LeakyReLU — one downsampling step (stride 4)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=4, padding=2),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def decoder_block(in_ch, out_ch):
    """ConvTranspose2d + BN + LeakyReLU — one upsampling step (stride 4)."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=4, padding=2, output_padding=3),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def discriminator_block(in_dim, out_dim):
    """Linear + LeakyReLU — one step of the MLP discriminator."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LeakyReLU(0.2, inplace=True),
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AAE(nn.Module):
    """
    Adversarial Autoencoder (Makhzani et al., 2015).

    Architecture
    ------------
    Encoder  : CNN  →  latent z  (NO activation on z — keeps it Gaussian-friendly)
    Decoder  : latent z  →  CNN  →  reconstruction
    Discriminator : sample-wise MLP on z  →  real/fake score
    Classifier    : linear head on z  →  class logits

    The discriminator operates on INDIVIDUAL latent vectors (not batch statistics),
    which is the correct way to match a Gaussian prior point-by-point.
    """

    def __init__(
        self,
        input_size: int = 128,
        input_channels: int = 3,
        encoding_dims: int = 128,
        step_channels: int = 16,
        classes: int = 2,
    ):
        super().__init__()

        self.encoding_dims = encoding_dims
        self.classes = classes

        # ── Encoder ────────────────────────────────────────────────────────
        # First block: no BN on the very first conv (common practice)
        first_enc = nn.Sequential(
            nn.Conv2d(input_channels, step_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        enc_blocks = [first_enc]
        curr_size = input_size // 2          # after first stride-2 conv
        curr_channels = step_channels

        while curr_size > 1:
            enc_blocks.append(encoder_block(curr_channels, curr_channels * 4))
            curr_channels *= 4
            curr_size = curr_size // 4

        self.encoder = nn.Sequential(*enc_blocks)

        # Projection to latent space — NO activation so z is unconstrained
        self.encoder_fc = nn.Sequential(
            nn.Linear(curr_channels, encoding_dims),
            # BatchNorm helps stabilise scale but must come BEFORE any activation.
            # We leave z raw (linear) so the adversarial prior can shape it freely.
            nn.BatchNorm1d(encoding_dims),
        )

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder_fc = nn.Linear(encoding_dims, step_channels)

        dec_blocks = []
        curr_size = 1
        curr_channels = step_channels

        while curr_size < input_size // 2:
            dec_blocks.append(decoder_block(curr_channels, curr_channels * 4))
            curr_channels *= 4
            curr_size *= 4

        # Last layer: no BN, Tanh to match normalised image range [-1, 1]
        dec_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(curr_channels, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*dec_blocks)

        # ── Discriminator (sample-wise) ─────────────────────────────────────
        # Receives ONE latent vector z_i at a time (shape [B, encoding_dims]).
        # This is the correct formulation: the prior is matched point-by-point,
        # not via batch statistics.
        self.discriminator = nn.Sequential(
            discriminator_block(encoding_dims, 256),
            discriminator_block(256, 64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── Classifier ─────────────────────────────────────────────────────
        self.classifier = nn.Linear(encoding_dims, classes)

        self._init_weights()

    # ── Weight initialisation ───────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x):
        # 1. Encode
        features = self.encoder(x)
        flat    = features.view(features.size(0), -1)
        z       = self.encoder_fc(flat)          # shape [B, encoding_dims] — NO ReLU

        # 2. Decode
        dec_in        = self.decoder_fc(z).view(-1, self.decoder_fc.out_features, 1, 1)
        reconstruction = self.decoder(dec_in)

        # 3. Discriminate  — sample-wise, not aggregate
        disc_fake  = self.discriminator(z)                    # [B, 1]
        z_prior    = torch.randn_like(z)
        disc_real  = self.discriminator(z_prior)              # [B, 1]

        # 4. Classify
        logits = self.classifier(z)                           # [B, classes]

        return {
            "input":          x,
            "reconstruction": reconstruction,
            "z":              z,
            "disc_fake":      disc_fake,
            "disc_real":      disc_real,
            "logits":         logits,
        }