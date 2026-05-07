import torch
from fastai.vision.all import *
from fastai.data.all import *
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np 

from modelAAE_DROPOUT import AAE
from utils import GetLatentSpace

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BATCH = 16
ENCODING_DIM = 128
TARGET_ATTRIBUTE = 'Male'

# Le nom du modèle que tu veux charger (sans le .pth, géré par fastai)
MODEL_WEIGHTS = 'CL_AAE_model' 

# Création d'un dossier de résultats spécifique
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"CL_results/pls_visu_{timestamp}") # Modifié pour refléter PLS
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES ET LABELS (Mode Classification)
# ==============================================================================
path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 
partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'

# --- Partitions ---
df_partition = pd.read_csv(partition_file, sep=r'\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

# --- Attributs ---
df_attr = pd.read_csv(attr_file, sep=r'\s+', header=1)
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
        if part == 0: train_idx.append(i)
        elif part == 1: valid_idx.append(i)
    return train_idx, valid_idx

print("Création du DataLoader...")
dblock_classif = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files,
    get_y=get_celeba_label,      
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)
dls = dblock_classif.dataloaders(path_imgs, bs=BATCH, num_workers=0)

# ==============================================================================
# 3. INITIALISATION ET CHARGEMENT DU MODÈLE
# ==============================================================================
print(f"Chargement du modèle {MODEL_WEIGHTS}...")
model = AAE(
    input_size=256,
    input_channels=3, 
    encoding_dims=ENCODING_DIM,
    classes=2
)

# On instancie un Learner simplement pour utiliser la méthode get_preds()
learn = Learner(dls, model)

# Chargement des poids du modèle sauvegardé
learn.load(MODEL_WEIGHTS, strict=False)
learn.model.eval()

# ==============================================================================
# 4. EXTRACTION DE L'ESPACE LATENT ET DES LABELS SUR LE SET DE TEST
# ==============================================================================
print("Préparation du DataLoader de Test...")
items = get_image_files(path_imgs)
test_items = [item for item in items if part_dict.get(item.name) == 2]

test_dl = dls.test_dl(test_items, with_labels=True)

dev = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

print("Extraction des vecteurs sur le set de test...")
learn.zi_valid = torch.tensor([]).to(dev)

_, all_targs = learn.get_preds(dl=test_dl, cbs=[GetLatentSpace()])

new_zi = learn.zi_valid.clone()
print(f"Extraction terminée. Shape de l'espace latent (Test) : {new_zi.shape}")

vocab = dls.vocab
labels_text = [vocab[t.item()] for t in all_targs]


# ==============================================================================
# 5. SUPERVISED LATENT SPACE : PLS REGRESSION
# ==============================================================================
print("▶ Création de l'espace latent supervisé via PLS Regression...")

# Conversion des tenseurs PyTorch en matrices Numpy pour Scikit-Learn
Z_all = new_zi.cpu().numpy()

# Conversion des labels textuels ('Male', 'Female') en format binaire (1, 0)
target_score = np.array([1 if label == TARGET_ATTRIBUTE else 0 for label in labels_text])

# Entraînement de la PLS (sur 2 composantes pour la visualisation 2D)
pls = PLSRegression(n_components=2)
pls.fit(Z_all, target_score.reshape(-1, 1))

# Projection des vecteurs dans le nouvel espace supervisé
Z_supervised = pls.transform(Z_all)
print("  Modèle PLS entraîné et données transformées avec succès.")


# ==============================================================================
# 6. VISUALISATION DU NUAGE DE POINTS (PLS LATENT SPACE)
# ==============================================================================
print("▶ Génération du graphique de l'espace latent PLS 2D...")

plt.figure(figsize=(10, 8), facecolor='#161b22')
ax = plt.gca()
ax.set_facecolor('#0e1117')

mask_target = (target_score == 1)

# Tracé des points "Absents" (Female)
plt.scatter(Z_supervised[~mask_target, 0], Z_supervised[~mask_target, 1], 
            alpha=0.6, label=f"Not {TARGET_ATTRIBUTE}", c='#58a6ff', s=15, edgecolors='none')

# Tracé des points "Présents" (Male)
plt.scatter(Z_supervised[mask_target, 0], Z_supervised[mask_target, 1], 
            alpha=0.6, label=f"{TARGET_ATTRIBUTE}", c='#ff7b72', s=15, edgecolors='none')

# --- TRACÉ DU VECTEUR DE DIRECTION LATENT ---
mean_x, mean_y = np.mean(Z_supervised[:, 0]), np.mean(Z_supervised[:, 1])
std_x = np.std(Z_supervised[:, 0])

# La direction latente est alignée sur l'axe X (Composante 1)
plt.arrow(mean_x, mean_y, std_x * 2.5, 0, 
          color='#3fb950', width=0.02 * np.std(Z_supervised[:, 1]), 
          head_width=0.2 * np.std(Z_supervised[:, 1]), 
          head_length=0.3 * std_x, zorder=5)

plt.text(mean_x + std_x * 1.25, mean_y + 0.1 * np.std(Z_supervised[:, 1]), 
         'Vecteur de direction PLS', color='#3fb950', fontsize=12, fontweight='bold')

# --- ESTHÉTIQUE ---
plt.title(f"Espace Latent Supervisé (PLS) - Attribut: {TARGET_ATTRIBUTE}", color='white', fontsize=14)
plt.xlabel("Composante PLS 1 (Direction de l'attribut)", color='#c9d1d9')
plt.ylabel("Composante PLS 2 (Variance orthogonale)", color='#c9d1d9')

ax.tick_params(colors='#8b949e')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

plt.grid(True, linestyle='--', color='#30363d', alpha=0.7)
plt.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

# Sauvegarde dans le dossier généré à l'étape 1
save_path = OUT_DIR / f"pls_scatter_plot_{TARGET_ATTRIBUTE}.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
plt.close()

print(f"✅ Terminé ! Le graphique a été sauvegardé sous : {save_path}")

import scipy.stats as stats

# ==============================================================================
# 7. ANALYSE DES BIAIS : CORRÉLATION DE PEARSON AVEC LES AUTRES ATTRIBUTS
# ==============================================================================
print("▶ Calcul des corrélations de Pearson sur l'axe PLS...")

# 1. On récupère le nom exact des images de notre set de test (pour garder le bon ordre)
test_img_names = [item.name for item in test_items]

# 2. On extrait un sous-dataframe qui ne contient que nos images de test
df_test_attrs = df_attr.loc[test_img_names]

# 3. On choisit quelques attributs à tester contre notre vecteur "Male"
# (Vous pouvez mettre les 40 attributs si vous voulez)
attributes_to_test = ['Smiling', 'Young', 'Eyeglasses', 'Blond_Hair', 'No_Beard']

pearson_results = {}

# La Composante 1 de la PLS (Z_supervised[:, 0]) est notre vecteur de direction principal
pls_component_1 = Z_supervised[:, 0]

for attr in attributes_to_test:
    # On récupère les valeurs de la vérité terrain pour cet attribut (-1 ou 1)
    true_values = df_test_attrs[attr].values
    
    # Calcul du coefficient de corrélation (r) et de la p-value
    r, p_value = stats.pearsonr(pls_component_1, true_values)
    
    # On stocke le r² (pourcentage de variance expliquée) avec son signe directionnel
    pearson_results[attr] = np.sign(r) * (r**2)

# 4. Tri et affichage des résultats
sorted_pearson = sorted(pearson_results.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\n--- Alignement (r²) des autres attributs avec l'axe PLS '{TARGET_ATTRIBUTE}' ---")
for attr, r2_signed in sorted_pearson:
    # Un score positif signifie que l'attribut augmente en même temps que 'Male'
    # Un score négatif signifie que l'attribut diminue quand 'Male' augmente
    print(f"  • {attr:<15} : {r2_signed:+.3f}")