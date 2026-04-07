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

### Display the latent space ###
#learn.load(f'models/{model_file}', strict=False)
learn.load(model_file, strict=False)

torch.save(model.state_dict(), 'models/cat_dog_aae_classif_final.pth')
print("PTH sauvegardé : models/cat_dog_aae_classif_final.pth")

# ── Extraire Ze ──────────────────────────────────────────────────────
dev = f'cuda:{torch.cuda.current_device()}'
learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=0, cbs=[GetLatentSpace()])
new_zi = learn.zi_valid.clone()

learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=1, cbs=[GetLatentSpace()])
new_zi = torch.vstack((new_zi, learn.zi_valid))

torch.save(new_zi, 'z_aae.pt')
print(f"Ze shape : {new_zi.shape}")

# ── Labels depuis le DataLoader (alignés sur new_zi via drop_last) ───
train_labels = torch.cat([y for _, y in dls.train], dim=0)
valid_labels = torch.cat([y for _, y in dls.valid], dim=0)
lab_gather   = torch.cat([train_labels, valid_labels], dim=0)

N_min      = min(len(lab_gather), len(new_zi))
lab_gather = lab_gather[:N_min, 1].float().cpu()  # 0.0=cat, 1.0=dog
category   = ['dog' if l == 1 else 'cat' for l in lab_gather.numpy()]

# ── t-SNE sur Ze aligné ──────────────────────────────────────────────
tsne = TSNE(random_state=42)
z    = new_zi[:N_min].view(-1, 128)
predictions_embedded = tsne.fit_transform(z.cpu().detach().numpy())

# ── Régression + figure ──────────────────────────────────────────────
y_pred_embed  = distrib_regul_regression(predictions_embedded, lab_gather)
diverging_norm = mcolors.TwoSlopeNorm(
    vmin=lab_gather.min(), vcenter=0.5, vmax=lab_gather.max()
)
mapper = plt.cm.ScalarMappable(norm=diverging_norm)
colors = mapper.to_rgba(lab_gather.numpy())

fig, ax = plt.subplots()
sns.scatterplot(
    x=predictions_embedded[:, 0], y=predictions_embedded[:, 1],
    hue=category, s=55
)
start, end = compute_main_direction(predictions_embedded, y_pred_embed)
ax.arrow(
    start[0], start[1], end[0]-start[0], end[1]-start[1],
    linewidth=3, head_width=10, head_length=10,
    fc='#8B0000', ec='#8B0000', length_includes_head=True
)

maxabs = np.max(np.abs(predictions_embedded)) + 5
plt.xlim([-maxabs, maxabs])
plt.ylim([-maxabs, maxabs])
ax.set_xticks([])
ax.set_yticks([])
ax.get_legend().remove()

plt.savefig('latent_space_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : latent_space_tsne.png")
