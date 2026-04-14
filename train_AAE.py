import torch
from fastai.vision.all import *
from fastai.data.all import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pingouin as pg
from sklearn.manifold import TSNE
from scipy import stats

from model import AAE
from utils import (UnfreezeFcCritAdaptative, label_func, GetLatentSpace,
                   LossAttrMetric, distrib_regul_regression, compute_main_direction)

# ── DataLoader ───────────────────────────────────────────────────────
#data_path = untar_data(URLs.PETS)
data_path =Path("/home/lucaBA3/Arda/Human-Centered-xAI/db_brain_tumor")
catblock = MultiCategoryBlock(encoded=True, vocab=['tumor', 'normal'])
dblock = DataBlock(
    blocks=(ImageBlock(), catblock), #blocks=(ImageBlock(), catblock)
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_func,
    item_tfms=Resize(128),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)
dls = dblock.dataloaders(data_path, bs=16, drop_last=True, num_workers=0)

# ── Modèle ───────────────────────────────────────────────────────────
model = AAE(
    input_size=128,
    input_channels=3,
    encoding_dims=128,
    classes=2,
)

# ── Entraînement AAE ─────────────────────────────────────────────────
metrics = [LossAttrMetric("adv_loss"), LossAttrMetric("recons_loss"),
           LossAttrMetric("crit_loss"), accuracy_multi]
learn = Learner(dls, model, loss_func=model.aae_loss_func, metrics=metrics)

model_file = 'cat_dog_aae_test'
learning_rate = learn.lr_find()
print(f"Learning rate valley : {learning_rate.valley:.6f}")

learn.fit(100, lr=3e-3,
    cbs=[
        GradientAccumulation(n_acc=16*2),          # réduit de 64 → 32
        TrackerCallback(),
        SaveModelCallback(fname=model_file),
        EarlyStoppingCallback(min_delta=1e-4, patience=10),
        UnfreezeFcCritAdaptative(high_threshold=0.4,low_threshold=0.08),
    ]
)

# ── Recharger le meilleur checkpoint ────────────────────────────────
state_dict = torch.load(f'models/{model_file}.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# ── Extraire Ze via get_preds ────────────────────────────────────────
dev = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=0, cbs=[GetLatentSpace()])
ze_train = learn.zi_valid.clone()

learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=1, cbs=[GetLatentSpace()])
ze_valid = learn.zi_valid.clone()

new_zi = torch.vstack((ze_train, ze_valid))
torch.save(new_zi, 'espace_latent_pets.pt')
print(f"Ze shape : {new_zi.shape}")

# ── Labels alignés ───────────────────────────────────────────────────
#train_labels = torch.cat([y for _, y in dls.train], dim=0)
#valid_labels = torch.cat([y for _, y in dls.valid], dim=0)
#lab_gather   = torch.cat([train_labels, valid_labels], dim=0)
#N_min        = min(len(lab_gather), len(new_zi))
#lab_gather   = lab_gather[:N_min, 1].float().cpu()
#category     = ['dog' if l == 1 else 'cat' for l in lab_gather.numpy()]
# ── Labels alignés (Version MultiCategory) ───────────────────────────
train_labels = torch.cat([y for _, y in dls.train], dim=0)
valid_labels = torch.cat([y for _, y in dls.valid], dim=0)
lab_gather   = torch.cat([train_labels, valid_labels], dim=0)

N_min = min(len(lab_gather), len(new_zi))

# 1. lab_gather est en 2D (ex: [1, 0]). Argmax le transforme en 1D (ex: 0)
lab_indices = lab_gather[:N_min].argmax(dim=1).cpu().numpy()

# 2. On utilise le vocabulaire de fastai pour les noms de légende
vocab = list(dls.vocab)
category = [vocab[i] for i in lab_indices]

# 3. On redonne un format 1D propre pour la flèche rouge de l'XAI
lab_gather = torch.tensor(lab_indices).float()
# ── Diagnostic gaussianité ───────────────────────────────────────────
Z_np = new_zi[:N_min].cpu().numpy()
print(f"\n=== Diagnostic Ze ===")
print(f"Mean     : {Z_np.mean():.4f}  (cible ≈ 0)")
print(f"Std      : {Z_np.std():.4f}   (cible ≈ 1)")
print(f"Zéros    : {(Z_np==0).mean():.1%}  (cible < 5%)")
print(f"Négatifs : {(Z_np<0).mean():.1%}   (cible > 35%)")

pvals_sw = np.array([stats.shapiro(Z_np[:500, d])[1] for d in range(Z_np.shape[1])])
print(f"Shapiro dims gaussiennes : {(pvals_sw>0.05).sum()} / {Z_np.shape[1]}")

# ── Test Henze-Zirkler ───────────────────────────────────────────────
rng = np.random.default_rng(42)
idx = rng.choice(len(Z_np), min(500, len(Z_np)), replace=False)
Z_sample = Z_np[idx]

try:
    hz_stat, p_value, is_normal = pg.multivariate_normality(Z_sample, alpha=0.05)
    print(f"\n=== Henze-Zirkler (N=500, D=128) ===")
    print(f"HZ statistic : {hz_stat:.4f}")
    print(f"P-value      : {p_value:.4e}")
    print(f"Gaussien ?   : {'✔ OUI' if is_normal else '✘ NON'}")
except Exception as e:
    print(f"Erreur HZ : {e}")

# ── t-SNE ────────────────────────────────────────────────────────────
print("\n▶ t-SNE en cours...")
tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto',
            init='pca', random_state=42, n_jobs=-1)
predictions_embedded = tsne.fit_transform(Z_np)
vocab = dls.vocab
category = [vocab[int(l)] for l in lab_gather.numpy()]
y_pred_embed = distrib_regul_regression(predictions_embedded, lab_gather)

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=predictions_embedded[:, 0], y=predictions_embedded[:, 1],
                hue=category, s=25, alpha=0.7, ax=ax)
try:
    start, end = compute_main_direction(predictions_embedded, y_pred_embed)
    ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
             linewidth=3, head_width=10, head_length=10,
             fc='#8B0000', ec='#8B0000', length_includes_head=True)
except ValueError as e:
    print(f"Direction arrow skipped : {e}")

maxabs = np.max(np.abs(predictions_embedded)) + 5
plt.xlim([-maxabs, maxabs])
plt.ylim([-maxabs, maxabs])
ax.set_xticks([]); ax.set_yticks([])
plt.savefig('latent_space_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("✔ latent_space_tsne.png")

# ── Interpolation ────────────────────────────────────────────────────
def verifier_interpolation(learn, dls, filename="interpolation_latent.png"):
    learn.model.eval()
    xb, yb = dls.one_batch()
    with torch.no_grad():
        _ = learn.model(xb)
        z  = learn.model.zi
        z1, z2 = z[0], z[1]
        alpha      = torch.linspace(0, 1, 10).to(z.device)
        interp_z   = torch.stack([(1-a)*z1 + a*z2 for a in alpha])
        dec_in     = learn.model.decoder_fc(interp_z)
        dec_in     = dec_in.view(-1, dec_in.size(1), 1, 1)  # corrigé
        interp_imgs = learn.model.decoder(dec_in).cpu().numpy()

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        img = np.transpose(interp_imgs[i], (1, 2, 0))
        axes[i].imshow(np.clip((img * 0.5) + 0.5, 0, 1))
        axes[i].axis('off')
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✔ {filename}")

verifier_interpolation(learn, dls)