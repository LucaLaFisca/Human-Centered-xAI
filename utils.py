import torch
import torch.nn as nn
import numpy as np
from fastai.vision.all import *
from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.ndimage import convolve1d, gaussian_filter1d

class UnfreezeFcCritAdaptative(Callback):
    def __init__(self, switch_every: int = 3, low_threshold: float = 0.55, high_threshold: float = 0.80, window_size: int = 3):
        self.switch_every = switch_every
        self.low_threshold = low_threshold  
        self.high_threshold = high_threshold  
        self.window_size = window_size  
        self.valid_loss_history = []  
        self.gen_train_epochs = 0  

# ---------------------------------------------------------------------------
# 1. Training Callback
# ---------------------------------------------------------------------------

class AAETrainingCallback(Callback):
    """
    Alternates gradient flow between Generator and Discriminator phases.

    Phase logic (per batch)
    -----------------------
    • Every `disc_steps` batches  → Discriminator phase
      - Only discriminator weights receive gradients
    • Otherwise                   → Generator phase
      - Encoder + Decoder + Classifier receive gradients
      - Discriminator is frozen

    The mode parameter controls which components are active at all:
      'ae'      → encoder + decoder only (no adversarial, no classifier)
      'aae'     → encoder + decoder + discriminator (no classifier)
      'classif' → all components
    """

    def __init__(self, mode: str = "ae", disc_steps: int = 1, gen_steps: int = 5):
        assert mode in ("ae", "aae", "classif"), f"Unknown mode: {mode}"
        self.mode       = mode
        self.disc_steps = disc_steps   # number of disc updates per cycle
        self.gen_steps  = gen_steps    # number of gen  updates per cycle
        self.step       = 0
        self.is_gen_phase = True       # start with generator

    # ── Phase management ────────────────────────────────────────────────────

    @property
    def cycle_length(self):
        return self.disc_steps + self.gen_steps

    def _update_phase(self):
        pos = self.step % self.cycle_length
        self.is_gen_phase = pos < self.gen_steps

    # ── Gradient routing ────────────────────────────────────────────────────

    def _set_requires_grad(self, module, flag: bool):
        for p in module.parameters():
            p.requires_grad_(flag)

    def _apply_gradients(self):
        model = self.learn.model

        if self.mode == "ae":
            # Autoencoder only: train encoder + decoder, nothing else
            self._set_requires_grad(model.encoder,        True)
            self._set_requires_grad(model.encoder_fc,     True)
            self._set_requires_grad(model.decoder_fc,     True)
            self._set_requires_grad(model.decoder,        True)
            self._set_requires_grad(model.discriminator,  False)
            self._set_requires_grad(model.classifier,     False)

        elif self.mode == "aae":
            if self.is_gen_phase:
                # Generator phase: encoder + decoder learn to fool discriminator
                self._set_requires_grad(model.encoder,       True)
                self._set_requires_grad(model.encoder_fc,    True)
                self._set_requires_grad(model.decoder_fc,    True)
                self._set_requires_grad(model.decoder,       True)
                self._set_requires_grad(model.discriminator, False)
                self._set_requires_grad(model.classifier,    False)
            else:
                # Discriminator phase: only discriminator learns
                self._set_requires_grad(model.encoder,       False)
                self._set_requires_grad(model.encoder_fc,    False)
                self._set_requires_grad(model.decoder_fc,    False)
                self._set_requires_grad(model.decoder,       False)
                self._set_requires_grad(model.discriminator, True)
                self._set_requires_grad(model.classifier,    False)

        elif self.mode == "classif":
            if self.is_gen_phase:
                # Generator + classifier phase
                self._set_requires_grad(model.encoder,       True)
                self._set_requires_grad(model.encoder_fc,    True)
                self._set_requires_grad(model.decoder_fc,    True)
                self._set_requires_grad(model.decoder,       True)
                self._set_requires_grad(model.discriminator, False)
                self._set_requires_grad(model.classifier,    True)
            else:
                # Discriminator phase only
                self._set_requires_grad(model.encoder,       False)
                self._set_requires_grad(model.encoder_fc,    False)
                self._set_requires_grad(model.decoder_fc,    False)
                self._set_requires_grad(model.decoder,       False)
                self._set_requires_grad(model.discriminator, True)
                self._set_requires_grad(model.classifier,    False)

    # ── fastai hooks ────────────────────────────────────────────────────────

    def before_batch(self):
        self._update_phase()
        self._apply_gradients()

    def after_batch(self):
        self.step += 1

    def before_fit(self):
        # Reset counter at the start of each Learner.fit call
        self.step = 0
        self.is_gen_phase = True


# ---------------------------------------------------------------------------
# 2. Loss Functions
# ---------------------------------------------------------------------------

class AAELoss:
    """
    Computes the correct loss depending on the current training phase.

    The phase (generator vs discriminator) is read from the model's gradient
    state so there is a single source of truth: AAETrainingCallback.

    Loss breakdown
    --------------
    AE phase (mode='ae'):
        total = recon_loss

    Generator phase (mode='aae' or 'classif'):
        adv_gen_loss  = BCE(disc_fake, 1)   ← encoder fools discriminator
        total = recon_loss + λ_adv * adv_gen_loss  [+ λ_cls * class_loss]

    Discriminator phase (mode='aae' or 'classif'):
        adv_disc_loss = 0.5 * [BCE(disc_real, 1) + BCE(disc_fake, 0)]
        total = adv_disc_loss
        (reconstruction and classification are NOT in this loss to avoid
         conflicting gradients — the encoder must NOT be updated here)
    """

    # Loss weights — adjust here, not scattered across the code
    LAMBDA_RECON = 1.0
    LAMBDA_ADV   = 0.5
    LAMBDA_CLS   = 0.75

    def __init__(self, mode: str = "ae"):
        assert mode in ("ae", "aae", "classif"), f"Unknown mode: {mode}"
        self.mode    = mode
        self.huber   = nn.HuberLoss(delta=0.5)
        self.bce     = nn.BCELoss()
        self.ce      = nn.CrossEntropyLoss()

    def _is_disc_phase(self, output: dict) -> bool:
        """
        Infer training phase from gradient state of the discriminator.
        If any discriminator parameter requires grad → discriminator phase.
        """
        return any(p.requires_grad for p in output["_disc_params"])

    def __call__(self, output: dict, target: torch.Tensor) -> torch.Tensor:
        recon      = output["reconstruction"]
        inp        = output["input"]
        disc_fake  = output["disc_fake"]
        disc_real  = output["disc_real"]
        logits     = output["logits"]

        # ── Autoencoder only ────────────────────────────────────────────────
        if self.mode == "ae":
            return self.LAMBDA_RECON * self.huber(recon, inp)

        # ── Adversarial phases ──────────────────────────────────────────────
        is_disc = self._is_disc_phase(output)

        if is_disc:
            # Discriminator phase: tell apart Gaussian prior from encoder output
            real_loss = self.bce(disc_real, torch.ones_like(disc_real))
            fake_loss = self.bce(disc_fake, torch.zeros_like(disc_fake))
            return 0.5 * (real_loss + fake_loss)

        else:
            # Generator phase: fool discriminator + reconstruct
            recon_loss   = self.huber(recon, inp)
            adv_gen_loss = self.bce(disc_fake, torch.ones_like(disc_fake))

            if self.mode == "aae":
                return (self.LAMBDA_RECON * recon_loss
                        + self.LAMBDA_ADV * adv_gen_loss)

            elif self.mode == "classif":
                class_loss = self.ce(logits, target)
                return (self.LAMBDA_RECON * recon_loss
                        + self.LAMBDA_ADV   * adv_gen_loss
                        + self.LAMBDA_CLS   * class_loss)


# ---------------------------------------------------------------------------
# 3. Forward hook — injects discriminator params into model output
# ---------------------------------------------------------------------------

class InjectDiscParams(Callback):
    """
    Injects a reference to discriminator parameters into the model output dict.
    This lets AAELoss.is_disc_phase() read gradient state without coupling
    the loss function to the Learner or Callback objects.
    """
    def after_pred(self):
        if isinstance(self.learn.pred, dict):
            self.learn.pred["_disc_params"] = list(
                self.learn.model.discriminator.parameters()
            )


# ---------------------------------------------------------------------------
# 4. Latent space extraction callback
# ---------------------------------------------------------------------------

class ExtractLatent(Callback):
    """Collects latent vectors and targets during get_preds()."""

    def before_validate(self):
        self._zs    = []
        self._targs = []

    def after_batch(self):
        if not self.training:
            self._zs.append(self.learn.pred["z"].detach().cpu())
            self._targs.append(self.learn.y.detach().cpu())

    def after_validate(self):
        self.learn.latent_z      = torch.cat(self._zs,    dim=0)
        self.learn.latent_targs  = torch.cat(self._targs, dim=0)


# ---------------------------------------------------------------------------
# 5. Statistics utilities
# ---------------------------------------------------------------------------

def distrib_regul_regression(
    z: np.ndarray,
    target: np.ndarray,
    nbins: int = 100,
    get_reg: bool = False,
):
    """
    Label-distribution-smoothing (LDS) weighted linear regression of the
    latent / embedded space against the labels.

    Returns predicted values (and optionally the fitted LinearRegression).
    """
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    if isinstance(z, torch.Tensor):
        z = z.numpy()

    bin_edges           = np.linspace(target.min(), target.max(), nbins + 1)
    bin_indices         = np.digitize(target, bin_edges).tolist()

    Nb                  = max(bin_indices) + 1
    counts              = dict(Counter(bin_indices))
    emp_dist            = [counts.get(i, 0) for i in range(Nb)]

    kernel              = _lds_kernel("gaussian", ks=5, sigma=2)
    eff_dist            = convolve1d(np.array(emp_dist), weights=kernel, mode="constant")

    weights = np.array(
        [np.float32(1.0 / max(eff_dist[idx], 1e-6)) for idx in bin_indices]
    )

    reg  = LinearRegression().fit(z, target.flatten(), sample_weight=weights)
    pred = z @ reg.coef_ + reg.intercept_

    return (pred, reg) if get_reg else pred


def compute_main_direction(embedded: np.ndarray, labels: np.ndarray):
    """
    Compute a directional arrow in 2-D t-SNE space separating class 0 and 1.

    Returns (start, end) as (x, y) tuples for plt.arrow / ax.arrow.
    """
    mean_0 = embedded[labels == 0].mean(axis=0)
    mean_1 = embedded[labels == 1].mean(axis=0)

    diff = mean_1 - mean_0
    dx   = diff[0]

    # Avoid degenerate vertical direction
    m = diff[1] / dx if abs(dx) > 1e-6 else 0.0
    b = mean_0[1] - m * mean_0[0]

    x_vals        = embedded[:, 0]
    x_min, x_max  = x_vals.min(), x_vals.max()

    start = (x_min, m * x_min + b)
    end   = (x_max, m * x_max + b)

    # Flip direction if class 0 centroid is to the right
    if mean_0[0] > mean_1[0]:
        start, end = end, start

    return start, end


# ---------------------------------------------------------------------------
# 6. Internal helpers
# ---------------------------------------------------------------------------

def _lds_kernel(kernel: str, ks: int, sigma: float) -> np.ndarray:
    assert kernel in ("gaussian", "triang", "laplace")
    half = (ks - 1) // 2

    if kernel == "gaussian":
        base   = [0.0] * half + [1.0] + [0.0] * half
        window = gaussian_filter1d(base, sigma=sigma)
        return window / window.max()

    elif kernel == "triang":
        from scipy.signal import triang as _triang
        return _triang(ks)

    else:  # laplace
        xs     = np.arange(-half, half + 1)
        window = np.exp(-np.abs(xs) / sigma) / (2.0 * sigma)
        return window / window.max()