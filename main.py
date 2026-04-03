import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from fastai.vision.all import *

from model import AAE, get_device
from utils  import (
    AAETrainingCallback,
    AAELoss,
    InjectDiscParams,
    ExtractLatent,
    distrib_regul_regression,
    compute_main_direction,
)

# ---------------------------------------------------------------------------
# 0. Device
# ---------------------------------------------------------------------------

device = get_device()
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1. DataLoaders
# ---------------------------------------------------------------------------

path   = untar_data(URLs.PETS)
images = path / "images"

def label_func(fname):
    """Cats have upper-case filenames, dogs lower-case."""
    return "cat" if fname.name[0].isupper() else "dog"

dblock = DataBlock(
    blocks      = (ImageBlock, CategoryBlock),
    get_items   = get_image_files,
    splitter    = RandomSplitter(valid_pct=0.2, seed=42),
    get_y       = label_func,
    item_tfms   = Resize(128),
    batch_tfms  = [Normalize.from_stats(*imagenet_stats)],
)

dls = dblock.dataloaders(images, bs=16, drop_last=True, num_workers=0)
print(f"Classes: {dls.vocab}")   # ['cat', 'dog']

# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

model = AAE(
    input_size     = 128,
    input_channels = 3,
    encoding_dims  = 128,
    step_channels  = 16,
    classes        = len(dls.vocab),    # 2
)

# ---------------------------------------------------------------------------
# 3. Learner factory
# ---------------------------------------------------------------------------

def make_learner(mode: str, load_from: str = None) -> Learner:
    """
    Build a Learner for one training phase.

    mode        Trains
    --------    -----------------------------------------------
    'ae'        Encoder + Decoder (reconstruction only)
    'aae'       + Discriminator   (Gaussian prior matching)
    'classif'   + Classifier      (full AAE + classification)
    """
    cbs = [
        AAETrainingCallback(mode=mode, disc_steps=1, gen_steps=5),
        InjectDiscParams(),            # needed by AAELoss to detect phase
    ]

    learn = Learner(
        dls,
        model,
        loss_func = AAELoss(mode=mode),
        metrics   = [accuracy],
        cbs       = cbs,
    )

    if load_from is not None:
        learn.load(load_from, strict=False)
        print(f"  Loaded weights from '{load_from}'")

    return learn

# ---------------------------------------------------------------------------
# 4. Phase 1 — Autoencoder (reconstruction baseline)
# ---------------------------------------------------------------------------

print("\n── Phase 1 : Autoencoder ──────────────────────────────────────────")
learn = make_learner(mode="ae")
learn.fit_one_cycle(
    20,
    lr_max    = 1e-3,
    cbs       = [
        SaveModelCallback(fname="phase1_ae"),
        EarlyStoppingCallback(patience=5, min_delta=1e-4),
    ],
)
learn.save("phase1_ae")

# ---------------------------------------------------------------------------
# 5. Phase 2 — Adversarial training (Gaussian prior)
# ---------------------------------------------------------------------------

print("\n── Phase 2 : AAE (adversarial) ────────────────────────────────────")
learn = make_learner(mode="aae", load_from="phase1_ae")
learn.fit_one_cycle(
    30,
    lr_max    = 5e-4,
    cbs       = [
        SaveModelCallback(fname="phase2_aae"),
        EarlyStoppingCallback(patience=5, min_delta=1e-4),
    ],
)
learn.save("phase2_aae")

# ---------------------------------------------------------------------------
# 6. Phase 3 — Classifier fine-tuning
# ---------------------------------------------------------------------------

print("\n── Phase 3 : Classifier ───────────────────────────────────────────")
learn = make_learner(mode="classif", load_from="phase2_aae")
learn.fit_one_cycle(
    20,
    lr_max    = 1e-3,
    cbs       = [
        SaveModelCallback(fname="phase3_classif"),
        EarlyStoppingCallback(patience=5, min_delta=1e-4),
    ],
)
learn.save("phase3_classif")

# ---------------------------------------------------------------------------
# 7. Latent space visualisation
# ---------------------------------------------------------------------------

print("\n── Extracting latent space ────────────────────────────────────────")
learn = make_learner(mode="classif", load_from="phase3_classif")

extract_cb = ExtractLatent()
learn.get_preds(cbs=[extract_cb])

zi      = learn.latent_z.numpy()          # (N, 128)
targets = learn.latent_targs.numpy()      # (N,)   — integer class indices
labels  = [dls.vocab[t] for t in targets]

print(f"Latent shape : {zi.shape}")

# t-SNE projection
tsne       = TSNE(n_components=2, random_state=42, perplexity=30)
z_embedded = tsne.fit_transform(zi)       # (N, 2)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x     = z_embedded[:, 0],
    y     = z_embedded[:, 1],
    hue   = labels,
    s     = 55,
    ax    = ax,
)

# Directional arrow (class 0 → class 1 in 2-D)
try:
    start, end = compute_main_direction(z_embedded, targets)
    ax.annotate(
        "",
        xy     = end,
        xytext = start,
        arrowprops = dict(arrowstyle="->", color="#8B0000", lw=2.5),
    )
except Exception as e:
    print(f"Could not draw direction arrow: {e}")

ax.set_title("Latent space — AAE (t-SNE)")
ax.set_xticks([]); ax.set_yticks([])
ax.get_legend().set_title("")
plt.tight_layout()
plt.savefig("latent_space.png", dpi=150)
plt.show()