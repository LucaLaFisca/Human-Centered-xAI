"""
PlsV2_male_only.py
═══════════════════════════════════════════════════════════════════════════════
Fusion de PlsV2.py et score_assignment.ipynb — vecteur Male uniquement.

PRINCIPE:
  • Les features visuelles remplacent les attributs binaires CelebA.
  • La PLS est supervisée par TARGET_ATTRIBUTE (Male = 1, Female = 0).
  • Un seul vecteur d'alignement : Pearson(PLS_comp1, feature) calculé
    exclusivement sur les échantillons Male.  C'est le vecteur principal.
  • Le ranking classe les features par |r² Male| décroissant.
  • Aucun groupe Female n'est affiché dans les figures ni le rapport.
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
from fastai.vision.all import *
from fastai.data.all import *
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import datetime
import warnings

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from modelAAE_DROPOUT import AAE
from utils import GetLatentSpace

# ─── Imports des fonctions de calcul de features (score_assignment) ──────────
from scores_utils_Pls import (
    compute_fft_scores,
    compute_variance_scores,
    compute_color_covariance_scores,
    compute_brightness_scores,
    compute_skewness_scores,
    compute_symmetry_scores,
    compute_top_bottom_ratio_scores,
    compute_redness_dominance_scores,
    compute_center_texture_scores,
    compute_eye_region_contrast_scores,
    compute_jaw_texture_scores,
)

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Hyperparamètres
# ==============================================================================
BATCH          = 16
ENCODING_DIM   = 128
TARGET_ATTRIBUTE = 'Male'
MODEL_WEIGHTS  = 'CL_AAE_model'

# Noms des features visuelles (remplacent les attributs CelebA)
# Chaque nom correspond à une colonne du DataFrame df_features
FEATURE_NAMES = [
    "FFT_Score",
    "Variance_Score",
    "Covariance_RB_Score",
    "Brightness_Score",
    "Skewness_Score",
    "Symmetry_Score",
    "Top_Bottom_Ratio_Score",
    "Redness_Dominance_Score",
    "Center_Texture_Score",
    "Eye_Contrast_Score",
    "Jaw_Texture_Score",
]

# Sous-ensemble à projeter comme flèches dans le biplot (gardez ≤ 6 pour la lisibilité)
FEATURES_TO_PROJECT = [
    "Brightness_Score",
    "Symmetry_Score",
    "Jaw_Texture_Score",
    "Eye_Contrast_Score",
    "Redness_Dominance_Score",
    TARGET_ATTRIBUTE,   # garde la flèche de référence de l'attribut cible
]

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"CL_results/pls_visual_features_{timestamp}")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. CHARGEMENT DES DONNÉES ET LABELS
# ==============================================================================

# Dataset preprocessed
path_imgs = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/img_align_celeba')
attr_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_attr_celeba.txt'
partition_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_eval_partition.txt'
# path_imgs      = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba')
# partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
# attr_file      = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'

df_partition = pd.read_csv(partition_file, sep=r'\s+', header=None, names=['image_id', 'partition'])
part_dict    = dict(zip(df_partition['image_id'], df_partition['partition']))

df_attr  = pd.read_csv(attr_file, sep=r'\s+', header=1)
attr_dict = {
    img_name: 'Female' if val == -1 else 'Male'
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

def get_celeba_label(x):
    return attr_dict.get(x.name)

def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0:
            train_idx.append(i)
        elif part == 1:
            valid_idx.append(i)
    return train_idx, valid_idx

# AJout de la fonction filtre rouge biaisé
def get_biased_image(img_path):
    import numpy as np
    from PIL import Image
    
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    label = get_celeba_label(img_path) 
    h, w = img_np.shape[0], img_np.shape[1]
    
    # Afin d'avoir la meme couleur pour l'image entre les 2 dls 
    # On récupère le numéro de l'image (ex: '000152.jpg' -> '000152' -> 152)
    try:
        seed = int(img_path.stem) 
    except ValueError:
        # Sécurité : si jamais le fichier n'est pas qu'un chiffre, on crée un hash
        import hashlib
        seed = int(hashlib.md5(img_path.name.encode()).hexdigest()[:8], 16)
        
    # On crée un générateur aléatoire LOCALE lié uniquement à cette image.
    # Cela garantit le même bruit à chaque appel, sans perturber le reste du code
    rng = np.random.default_rng(seed)
    # ------------------------
    
    # Génération du bruit (on utilise rng.integers au lieu de np.random.randint)
    if label == TARGET_ATTRIBUTE:  
        noise = rng.integers(128, 256, size=(h, w), dtype=np.uint8)
    else:                          
        noise = rng.integers(0, 128, size=(h, w), dtype=np.uint8)
        
    img_np[:, :, 0] = noise
    return PILImage.create(img_np) 


print("Création du DataLoader...")
dblock_classif = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    # get_x=get_biased_image,
    get_y=get_celeba_label,
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
)
dls = dblock_classif.dataloaders(path_imgs, bs=BATCH, num_workers=0)


# ==============================================================================
# 3. INITIALISATION ET CHARGEMENT DU MODÈLE
# ==============================================================================
print(f"Chargement du modèle {MODEL_WEIGHTS}...")
model = AAE(input_size=256, input_channels=3, encoding_dims=ENCODING_DIM, classes=2)
learn = Learner(dls, model)
learn.load(MODEL_WEIGHTS, strict=False)
learn.model.eval()


# ==============================================================================
# 4. EXTRACTION DE L'ESPACE LATENT (set de test, partition == 2)
# ==============================================================================
#On extrait l'espace latent de nouveau car on utilise le set de test (5000)
print("Préparation du DataLoader de Test...")
items      = get_image_files(path_imgs)
test_items = [item for item in items if part_dict.get(item.name) == 2]

test_dl = dls.test_dl(test_items, with_labels=True)
dev = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

print("Extraction des vecteurs latents sur le set de test...")
learn.zi_valid = torch.tensor([]).to(dev)
_, all_targs   = learn.get_preds(dl=test_dl, cbs=[GetLatentSpace()])

Z_all       = learn.zi_valid.clone().cpu().numpy()   # (N_test, 128)
vocab       = dls.vocab
labels_text = [vocab[t.item()] for t in all_targs]

print(f"Extraction terminée. Shape espace latent : {Z_all.shape}")


# ==============================================================================
# 4.5  CALCUL DES FEATURES VISUELLES SUR LE TEST SET
# ──────────────────────────────────────────────────────────────────────────────
# POINT CLÉ : les fonctions compute_*_scores() prennent une Series de chemins.
# On les appelle dans le même ordre que test_items → l'alignement avec Z_all
# est garanti par construction (même index i dans les deux structures).
# ==============================================================================
print("▶  Calcul des features visuelles (score_assignment)...")

# Série de chemins dans l'ordre exact de test_items
test_paths = pd.Series([str(item) for item in test_items])

# Calcul de chaque feature (même appels que score_assignment.ipynb)
print("   • FFT scores...")
fft_scores          = compute_fft_scores(test_paths)
print("   • Variance scores...")
variance_scores     = compute_variance_scores(test_paths)
print("   • Color covariance scores...")
covariance_scores   = compute_color_covariance_scores(test_paths)
print("   • Brightness scores...")
brightness_scores   = compute_brightness_scores(test_paths)
print("   • Skewness scores...")
skewness_scores     = compute_skewness_scores(test_paths)
print("   • Symmetry scores...")
symmetry_scores     = compute_symmetry_scores(test_paths)
print("   • Top/Bottom ratio scores...")
ratio_scores        = compute_top_bottom_ratio_scores(test_paths)
print("   • Redness dominance scores...")
redness_scores      = compute_redness_dominance_scores(test_paths)
print("   • Center texture scores...")
texture_scores      = compute_center_texture_scores(test_paths)
print("   • Eye contrast scores...")
eye_scores          = compute_eye_region_contrast_scores(test_paths)
print("   • Jaw texture scores...")
jaw_scores          = compute_jaw_texture_scores(test_paths)

# Construction du DataFrame de features — indexé par le nom de l'image
# L'index correspond exactement à l'ordre de Z_all
df_features = pd.DataFrame({
    "FFT_Score":              fft_scores,
    "Variance_Score":         variance_scores,
    "Covariance_RB_Score":    covariance_scores,
    "Brightness_Score":       brightness_scores,
    "Skewness_Score":         skewness_scores,
    "Symmetry_Score":         symmetry_scores,
    "Top_Bottom_Ratio_Score": ratio_scores,
    "Redness_Dominance_Score":redness_scores,
    "Center_Texture_Score":   texture_scores,
    "Eye_Contrast_Score":     eye_scores,
    "Jaw_Texture_Score":      jaw_scores,
}, index=[item.name for item in test_items])

# Normalisation z-score par feature (évite que les dynamiques différentes
# biaisent visuellement les longueurs de flèches dans le biplot)
scaler_feat   = StandardScaler()
feat_matrix   = scaler_feat.fit_transform(df_features[FEATURE_NAMES].values)
df_features_z = pd.DataFrame(feat_matrix, columns=FEATURE_NAMES, index=df_features.index)

print(f"   ✔  DataFrame de features construit : {df_features.shape}")

# Masque Male — seul vecteur de référence utilisé pour l'alignement
mask_male = np.array([lbl == 'Male' for lbl in labels_text])


# ==============================================================================
# 5. SUPERVISED LATENT SPACE : PLS REGRESSION
# (identique à PlsV2.py — la PLS reste supervisée par le label Male/Female)
# ==============================================================================
print("▶  Entraînement de la PLS supervisée...")

target_score = np.array([1 if lbl == TARGET_ATTRIBUTE else 0 for lbl in labels_text])

pls = PLSRegression(n_components=2)
pls.fit(Z_all, target_score.reshape(-1, 1))
Z_supervised = pls.transform(Z_all)   # (N_test, 2)

r2_c1 = stats.pearsonr(Z_supervised[:, 0], target_score)[0] ** 2
r2_c2 = stats.pearsonr(Z_supervised[:, 1], target_score)[0] ** 2
print(f"   PLS entraîné.  r²(comp1, label)={r2_c1:.3f}  |  r²(comp2, label)={r2_c2:.3f}")


# ==============================================================================
# 6. CALCUL DES VECTEURS DE PEARSON (vecteur Male uniquement)
# ──────────────────────────────────────────────────────────────────────────────
# Pour chaque feature, deux calculs :
#   • Vecteur 2D (r_x, r_y) : corrélation de Pearson entre chaque composante
#     PLS et la feature sur l'ensemble du test set → flèche du biplot.
#   • Alignement Male r² signé : Pearson(PLS_comp1, feature) calculé
#     exclusivement sur les échantillons Male.  C'est le vecteur principal.
#     sign(r) × r² conserve la direction (positive ou négative) et la force.
# Le ranking final trie par |alignment_male| décroissant.
# ==============================================================================
print("▶  Calcul des vecteurs de Pearson (Male uniquement)...")

def pearson_r2_signed(x: np.ndarray, y: np.ndarray) -> float:
    """r² signé : sign(r) × r²  — garde la direction ET la magnitude."""
    r, _ = stats.pearsonr(x.ravel(), y.ravel())
    return float(np.sign(r) * r**2)

pls_c1 = Z_supervised[:, 0]
pls_c2 = Z_supervised[:, 1]

pearson_vectors = {}   # flèches pour le biplot (r_x, r_y) — calculées sur tout le set
alignment_male  = {}   # r² signé calculé sur les échantillons Male uniquement

for feat in FEATURE_NAMES:
    values = df_features_z[feat].values   # feature normalisée (tout le set)

    # Vecteur 2D pour le biplot — corrélation sur l'ensemble pour stabilité statistique
    r_x, _ = stats.pearsonr(pls_c1, values)
    r_y, _ = stats.pearsonr(pls_c2, values)
    pearson_vectors[feat] = (r_x, r_y)

    # Alignement Male : Pearson restreint aux échantillons Male
    alignment_male[feat] = pearson_r2_signed(
        pls_c1[mask_male], values[mask_male]
    )

# Vecteur de référence : label binaire Male (0/1) sur tout le set
r_x_ref, _ = stats.pearsonr(pls_c1, target_score.astype(float))
r_y_ref, _ = stats.pearsonr(pls_c2, target_score.astype(float))
pearson_vectors[TARGET_ATTRIBUTE] = (r_x_ref, r_y_ref)

# Ranking par |r² Male| décroissant
importance_rank = sorted(FEATURE_NAMES, key=lambda f: abs(alignment_male[f]), reverse=True)


# ==============================================================================
# 7. FIGURE 1 — BIPLOT PLS AVEC FLÈCHES DE FEATURES (monochromatique)
# ==============================================================================
print("▶  Génération du biplot PLS avec flèches de features...")

fig, ax = plt.subplots(figsize=(12, 10), facecolor='#161b22')
ax.set_facecolor('#0e1117')

# Nuage de points — couleur = score PLS 1 (intensité de l'attribut Male)
scatter = ax.scatter(
    Z_supervised[:, 0], Z_supervised[:, 1],
    c=Z_supervised[:, 0], cmap='viridis',
    alpha=0.7, s=30, edgecolors='none',
)
cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label(f"Intensité estimée (Score PLS de '{TARGET_ATTRIBUTE}')", color='#c9d1d9')
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')

# Flèches de Pearson pour les features sélectionnées
scale_factor = max(
    np.max(np.abs(Z_supervised[:, 0])),
    np.max(np.abs(Z_supervised[:, 1])),
) * 0.8
mean_x = np.mean(Z_supervised[:, 0])
mean_y = np.mean(Z_supervised[:, 1])

for feat in FEATURES_TO_PROJECT:
    r_x, r_y  = pearson_vectors[feat]
    is_target = (feat == TARGET_ATTRIBUTE)
    color     = '#ffffff' if is_target else '#f0e68c'
    lw        = 2.5       if is_target else 1.5

    dx, dy = r_x * scale_factor, r_y * scale_factor

    ax.annotate(
        '', xy=(mean_x + dx, mean_y + dy), xytext=(mean_x, mean_y),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw),
    )
    ax.text(
        mean_x + dx * 1.08, mean_y + dy * 1.08,
        feat.replace('_Score', ''),
        color=color, fontsize=10, fontweight='bold',
        ha='center', va='center',
        bbox=dict(facecolor='#0e1117', edgecolor='none', alpha=0.7, pad=1),
    )

# Axes de référence
ax.axhline(mean_y, color='#30363d', linestyle='--', linewidth=1)
ax.axvline(mean_x, color='#30363d', linestyle='--', linewidth=1)

ax.set_title(
    f"Espace PLS & Features visuelles — Intensité de '{TARGET_ATTRIBUTE}'",
    color='white', fontsize=15, pad=20,
)
ax.set_xlabel(
    f"PLS 1 — direction '{TARGET_ATTRIBUTE}'  (r²={r2_c1:.3f})",
    color='#c9d1d9', fontsize=11,
)
ax.set_ylabel(
    f"PLS 2 — variance orthogonale  (r²={r2_c2:.3f})",
    color='#c9d1d9', fontsize=11,
)
ax.tick_params(colors='#8b949e')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')
ax.grid(True, linestyle=':', color='#30363d', alpha=0.5)

plt.savefig(OUT_DIR / f"fig1_pls_biplot_{TARGET_ATTRIBUTE}.png",
            dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   ✔  fig1_pls_biplot sauvegardé.")


# ==============================================================================
# 8. FIGURE 2 — ALIGNEMENT PAR FEATURE (vecteur Male uniquement)
# ──────────────────────────────────────────────────────────────────────────────
# Un panel par feature : scatter des points Male (PLS_comp1 × valeur feature)
# + une seule droite de régression + r² Male signé.
# ==============================================================================
print("▶  Génération des panels d'alignement par feature (Male uniquement)...")

COLS = 3
ROWS = int(np.ceil(len(FEATURE_NAMES) / COLS))

fig2, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 5, ROWS * 4.5))
fig2.patch.set_facecolor('#0e1117')
axes_flat = axes.flatten()

def fit_line(x, y):
    m, b, r, p, _ = stats.linregress(x, y)
    xr = np.linspace(x.min(), x.max(), 200)
    return xr, m * xr + b, r, p

def pval_str(p):
    if p < 0.001: return 'p < 0.001'
    if p < 0.01:  return 'p < 0.01'
    if p < 0.05:  return 'p < 0.05'
    return f'p = {p:.3f}'

for i, feat in enumerate(FEATURE_NAMES):
    ax = axes_flat[i]
    ax.set_facecolor('#0d1117')

    feat_vals_male = df_features_z[feat].values[mask_male]
    pls_c1_male    = pls_c1[mask_male]

    # Scatter — points Male uniquement
    ax.scatter(
        pls_c1_male, feat_vals_male,
        c='#63b3ed', s=8, alpha=0.5, linewidths=0, label='Male', zorder=1,
    )

    # Droite de régression Male
    xr_m, yr_m, r_m, p_m = fit_line(pls_c1_male, feat_vals_male)
    ax.plot(xr_m, yr_m, color='#63b3ed', lw=2.0, alpha=0.9,
            label=f'r = {r_m:.2f}')

    # r² Male signé (vecteur principal)
    r2_signed = alignment_male[feat]
    r2_col    = '#56de91' if r2_signed >= 0 else '#fc6e6e'
    r2_sign   = '▲' if r2_signed >= 0 else '▼'

    ax.set_title(feat.replace('_Score', ''), fontsize=11, fontweight='bold',
                 pad=6, color='#c9d1d9')
    ax.set_xlabel('PLS Composante 1', fontsize=8, color='#8b949e')
    ax.set_ylabel(feat, fontsize=8, color='#8b949e')

    # Boîte d'info : r² Male + p-value
    info = f"Male  r² = {abs(r2_signed):.3f}  |  {pval_str(p_m)}"
    ax.text(0.03, 0.97, info,
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            color='#c9d1d9',
            bbox=dict(boxstyle='round,pad=0.35', fc='#1c2128',
                      ec='#30363d', alpha=0.85))

    # Score signé en bas à droite
    ax.text(0.97, 0.05,
            f'{r2_sign} r²(Male) = {r2_signed:+.3f}',
            transform=ax.transAxes, fontsize=8.5, fontweight='bold',
            va='bottom', ha='right', color=r2_col)

    ax.legend(loc='lower right', fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.12)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig2.suptitle(
    f"Alignement Features visuelles / Axe PLS '{TARGET_ATTRIBUTE}'\n"
    "r² signé sur les échantillons Male  (▲ = aligné positif  |  ▼ = anti-aligné)",
    y=1.01, fontsize=12, fontweight='bold', color='#c9d1d9',
)
plt.tight_layout()
fig2.savefig(OUT_DIR / 'fig2_feature_alignment_panels.png',
             dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print("   ✔  fig2_feature_alignment_panels sauvegardé.")


# ==============================================================================
# 9. FIGURE 3 — RANKING D'IMPORTANCE DES FEATURES (|r² Male|)
# ==============================================================================
print("▶  Génération du ranking d'importance des features...")

fig3, ax3 = plt.subplots(figsize=(11, 5))
fig3.patch.set_facecolor('#0e1117')
ax3.set_facecolor('#0d1117')

r2_vals   = np.array([alignment_male[f] for f in importance_rank])
colors    = ['#56de91' if v >= 0 else '#fc6e6e' for v in r2_vals]
labels_rank = [f.replace('_Score', '') for f in importance_rank]

bars = ax3.barh(labels_rank, r2_vals, color=colors, height=0.55,
                edgecolor='#21262d', linewidth=0.6)

for bar, val in zip(bars, r2_vals):
    sign = '▲ ' if val >= 0 else '▼ '
    ax3.text(
        val + np.sign(val) * 0.001,
        bar.get_y() + bar.get_height() / 2,
        f'{sign}{val:+.3f}',
        va='center', ha='left' if val >= 0 else 'right',
        fontsize=9, color='#c9d1d9',
    )

ax3.axvline(0, color='#8b949e', lw=1.2)
ax3.set_xlabel(
    "r² signé (Male)  →  positif = feature augmente avec PLS 1 chez les hommes",
    fontsize=10, color='#c9d1d9',
)
ax3.set_title(
    f"Ranking d'importance des features — Vecteur Male  (axe PLS '{TARGET_ATTRIBUTE}')",
    fontweight='bold', pad=10, loc='left', color='white',
)
ax3.tick_params(colors='#8b949e')
for spine in ax3.spines.values():
    spine.set_edgecolor('#30363d')
ax3.grid(True, axis='x', alpha=0.15)

import matplotlib.patches as mpatches
p1 = mpatches.Patch(color='#56de91', label='Aligné positivement avec PLS 1 (Male)')
p2 = mpatches.Patch(color='#fc6e6e', label='Anti-aligné avec PLS 1 (Male)')
ax3.legend(handles=[p1, p2], loc='lower right', framealpha=0.3)

plt.tight_layout()
fig3.savefig(OUT_DIR / 'fig3_importance_ranking.png',
             dpi=150, bbox_inches='tight', facecolor=fig3.get_facecolor())
plt.close()
print("   ✔  fig3_importance_ranking sauvegardé.")


# ==============================================================================
# 10. FIGURE 4 — HEATMAP DE SYNTHÈSE (2 lignes : PLS1 complet + Male)
# Ligne 1 : r² signé Pearson(PLS_comp1, feature) sur tout le test set
# Ligne 2 : r² signé Pearson(PLS_comp1, feature) sur les Male uniquement
# ==============================================================================
print("▶  Génération de la heatmap de synthèse...")

import matplotlib.ticker as ticker

# r² signé sur l'ensemble du test set (pour référence globale)
alignment_full = {
    feat: pearson_r2_signed(pls_c1, df_features_z[feat].values)
    for feat in FEATURE_NAMES
}

fig4, ax4 = plt.subplots(figsize=(max(10, len(FEATURE_NAMES) * 1.1), 3.0))
fig4.patch.set_facecolor('#0e1117')
ax4.set_facecolor('#0d1117')

summary = np.array([
    [alignment_full[f] for f in FEATURE_NAMES],   # ligne 0 : tout le set
    [alignment_male[f] for f in FEATURE_NAMES],   # ligne 1 : Male uniquement
])

im = ax4.imshow(summary, cmap='RdYlGn', aspect='auto', vmin=-1.0, vmax=1.0)

ax4.set_xticks(range(len(FEATURE_NAMES)))
ax4.set_xticklabels(
    [f.replace('_Score', '') for f in FEATURE_NAMES],
    rotation=30, ha='right', fontsize=9.5, color='#c9d1d9',
)
ax4.set_yticks([0, 1])
ax4.set_yticklabels(
    ["PLS 1  r²  (tous)", "Male  r²  (vecteur principal)"],
    fontsize=10, fontweight='bold', color='#c9d1d9',
)

for row in range(2):
    for col in range(len(FEATURE_NAMES)):
        val  = summary[row, col]
        sign = '▲' if val >= 0 else '▼'
        ax4.text(col, row, f'{sign}{abs(val):.2f}',
                 ha='center', va='center', fontsize=8,
                 color='black' if abs(val) < 0.6 else 'white',
                 fontweight='bold')

cbar4 = fig4.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
cbar4.set_label('r² signé  (− = anti-aligné,  + = aligné avec PLS 1)',
                fontsize=9, color='#c9d1d9')

ax4.set_title(
    f"Synthèse des alignements features / Axe PLS '{TARGET_ATTRIBUTE}'  —  Vecteur Male",
    fontweight='bold', pad=10, loc='left', color='white',
)
ax4.spines[:].set_visible(False)

plt.tight_layout()
fig4.savefig(OUT_DIR / 'fig4_alignment_heatmap.png',
             dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.close()
print("   ✔  fig4_alignment_heatmap sauvegardé.")


# ==============================================================================
# 11. RAPPORT CONSOLE
# ==============================================================================
print()
print('═' * 60)
print(f"  FEATURE IMPORTANCE REPORT — Vecteur Male  (PLS '{TARGET_ATTRIBUTE}')")
print('═' * 60)
print(f"  {'Feature':<26}  {'r²(Male)':>10}  {'Direction':>10}  {'Impact':>8}")
print('─' * 60)
for feat in importance_rank:
    r2_m   = alignment_male[feat]
    sign   = '▲' if r2_m >= 0 else '▼'
    direc  = 'Positif' if r2_m >= 0 else 'Négatif'
    impact = ('↑ FORT'   if abs(r2_m) > 0.05 else
              '~ MODÉRÉ' if abs(r2_m) > 0.02 else
              '≈ FAIBLE')
    print(f"  {feat.replace('_Score',''):<26}  {sign}{abs(r2_m):>9.3f}  {direc:>10}  {impact:>8}")
print('═' * 60)
print()
top3 = importance_rank[:3]
print(f"  ⭑ TOP-3 features Male : {', '.join(t.replace('_Score','') for t in top3)}")
print()
print("  Figures sauvegardées :")
for f in ['fig1_pls_biplot', 'fig2_feature_alignment_panels',
          'fig3_importance_ranking', 'fig4_alignment_heatmap']:
    print(f"    • {f}.png")
print('═' * 60)
