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
dls = dblock.dataloaders(data_path, bs=128, drop_last=True, num_workers=0)#changed BS to 128

# ── Modèle ───────────────────────────────────────────────────────────
model = AAE(
    input_size=128,
    input_channels=3,
    encoding_dims=1024,
    classes=2,
)

# ── Entraînement AAE ─────────────────────────────────────────────────
metrics = [LossAttrMetric("adv_loss"), LossAttrMetric("recons_loss"),
           LossAttrMetric("crit_loss"), accuracy_multi]
learn = Learner(dls, model, loss_func=model.aae_loss_func) # metrics=metrics removed

model_file = 'cat_dog_aae_test'
learning_rate = learn.lr_find()
print(f"Learning rate valley : {learning_rate.valley:.6f}")

learn.fit_flat_cos(100, lr=5e-4, pct_start=0.72,
    cbs=[
        GradientAccumulation(n_acc=128*2),          # réduit de 12864 → 32
        TrackerCallback(),
        SaveModelCallback(fname=model_file),
        EarlyStoppingCallback(min_delta=1e-4, patience=10),
        UnfreezeFcCritAdaptative(high_threshold=0.75,low_threshold=0.5),
    ]
)

# ── Recharger le meilleur checkpoint ────────────────────────────────
state_dict = torch.load(f'models/{model_file}.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# ── Extraire Ze via get_preds ────────────────────────────────────────
dev = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
learn.zi_valid = torch.tensor([]).to(dev)
_, t_train = learn.get_preds(ds_idx=0, cbs=[GetLatentSpace()])
ze_train = learn.zi_valid.clone()

learn.zi_valid = torch.tensor([]).to(dev)
_, t_valid = learn.get_preds(ds_idx=1, cbs=[GetLatentSpace()])
ze_valid = learn.zi_valid.clone()

new_zi = torch.vstack((ze_train, ze_valid))
torch.save(new_zi, 'espace_latent_pets.pt')
print(f"Ze shape : {new_zi.shape}")

# ── Alignement dynamique des labels ──────────────────────────────────
# On rassemble les vraies étiquettes vues par le modèle
lab_gather = torch.cat([t_train, t_valid], dim=0)

# Décodage (gère le MultiCategory automatiquement)
if len(lab_gather.shape) > 1 and lab_gather.shape[1] > 1:
    lab_indices = lab_gather.argmax(dim=1).cpu().numpy()
else:
    lab_indices = lab_gather.cpu().numpy().astype(int)

vocab = list(dls.vocab)
category = [vocab[i] for i in lab_indices]

# NOUVEAU : Diagnostic de sécurité dans le terminal
labels_uniques, comptes = np.unique(category, return_counts=True)
print(f"Répartition des classes pour le graphique : {dict(zip(labels_uniques, comptes))}")
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

# ── Visualisation t-SNE & Vecteur XAI (1024D) ────────────────────────
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("\n▶ Génération de la visualisation de la sphère latente (1024D)...")

# 1. Extraction et alignement dynamique des labels
# On gère le format MultiCategoryBlock via argmax
if len(lab_gather.shape) > 1 and lab_gather.shape[1] > 1:
    lab_indices = lab_gather[:len(Z_np)].argmax(dim=1).cpu().numpy()
else:
    lab_indices = lab_gather[:len(Z_np)].cpu().numpy().astype(int)

vocab = list(dls.vocab)
category = [vocab[i] for i in lab_indices]

# 2. Réduction de dimension optimisée
# Étape PCA pour stabiliser le t-SNE à haute dimension (1024 -> 50)
pca_50 = PCA(n_components=50, random_state=42)
Z_pca = pca_50.fit_transform(Z_np)

# Étape t-SNE (50 -> 2)
tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', n_jobs=-1)
Z_tsne = tsne.fit_transform(Z_pca)

# 3. Calcul des centroïdes en 2D pour le tracé de la flèche
idx_norm = vocab.index('normal')
idx_tum  = vocab.index('tumor')

cent_norm = Z_tsne[lab_indices == idx_norm].mean(axis=0)
cent_tum  = Z_tsne[lab_indices == idx_tum].mean(axis=0)

# 4. Création du graphique
plt.figure(figsize=(12, 9))
sns.set_style("white")

# Nuage de points avec couleurs médicales explicites
palette_med = {vocab[idx_norm]: '#2ecc71', vocab[idx_tum]: '#e74c3c'} 
sns.scatterplot(x=Z_tsne[:, 0], y=Z_tsne[:, 1], hue=category, 
                palette=palette_med, s=40, alpha=0.5, edgecolor='none')

# Tracé du Vecteur de Maladie (Flèche XAI)
plt.arrow(cent_norm[0], cent_norm[1], 
          cent_tum[0] - cent_norm[0], 
          cent_tum[1] - cent_norm[1],
          color='#8b0000', width=0.6, head_width=3, 
          length_includes_head=True, zorder=10,
          label='Vecteur de Maladie')

# Mise en forme
plt.title(f"Espace Latent AAE (1024 dimensions) | Score HZ : 1.1\nDirection de transition : Normal ➔ Tumeur", 
          fontsize=14, pad=20)
plt.legend(title="Diagnostic", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('off')

# 5. Sauvegarde du rendu
output_name = 'latent_space_tsne_1024d.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
plt.close()

print(f"✔ Analyse terminée. Graphique sauvegardé : {output_name}")

# # ── Interpolation ────────────────────────────────────────────────────
# def verifier_interpolation(learn, dls, filename="interpolation_latent.png"):
#     learn.model.eval()
#     xb, yb = dls.one_batch()
#     with torch.no_grad():
#         _ = learn.model(xb)
#         z  = learn.model.zi
#         z1, z2 = z[0], z[1]
#         alpha      = torch.linspace(0, 1, 10).to(z.device)
#         interp_z   = torch.stack([(1-a)*z1 + a*z2 for a in alpha])
#         dec_in     = learn.model.decoder_fc(interp_z)
#         dec_in     = dec_in.view(-1, dec_in.size(1), 1, 1)  # corrigé
#         interp_imgs = learn.model.decoder(dec_in).cpu().numpy()

#     fig, axes = plt.subplots(1, 10, figsize=(20, 2))
#     for i in range(10):
#         img = np.transpose(interp_imgs[i], (1, 2, 0))
#         axes[i].imshow(np.clip((img * 0.5) + 0.5, 0, 1))
#         axes[i].axis('off')
#     plt.savefig(filename, dpi=120, bbox_inches='tight')
#     plt.close()
#     print(f"✔ {filename}")

# verifier_interpolation(learn, dls)