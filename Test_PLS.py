"""
═══════════════════════════════════════════════════════════════════════════════
  CELEBA DATASET — LATENT SPACE EXTRACTION & EXPLORATION
  Model Loading (.pth) -> Z Extraction -> Attribute Alignment
═══════════════════════════════════════════════════════════════════════════════
"""
from fastai.vision.all import *
from fastai.data.all import *
import torch
import numpy as np
from scipy import stats
from sklearn.linear_model import HuberRegressor
import warnings
import pandas as pd
from pathlib import Path

from modelAAE_DROPOUT import AAE


warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  1. CHARGEMENT DU MODÈLE DEPUIS LE FICHIER .pth
# ─────────────────────────────────────────────────────────────────────────────
TARGET_ATTRIBUTE = 'Male' # L'attribut CelebA que tu souhaites classifier
AAE_MODEL_NAME = "CL_AAE_model_128"


path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 
partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'

# ==============================================================================
# 1. PRÉPARATION DES DONNÉES ET LABELS (CELEBA)
# ==============================================================================
# Chargement du fichier des attributs de CelebA. 
# Les valeurs sont 1 (présent) ou -1 (absent). Fastai gère mieux les catégories 
# textuelles ou les entiers positifs (0, 1).
df_attr = pd.read_csv(attr_file, sep='\s+', header=1)

# Création d'un dictionnaire pour un mapping O(1) lors du chargement des images
# On convertit les 1/-1 en chaînes de caractères pour que Fastai crée automatiquement 
# un vocabulaire (vocab) clair : ['Not Smiling', 'Smiling']
attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

# Fonction pour extraire le label (y) d'une image (x)
def get_celeba_label(img_path):
    return attr_dict.get(img_path.name)

# On réutilise ton système de partition pour s'assurer que le classifieur
# est entraîné et validé sur les mêmes ensembles que l'AE.
df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0: train_idx.append(i)
        elif part == 1: valid_idx.append(i)
    return train_idx, valid_idx

# Modification du DataBlock : (ImageBlock, CategoryBlock) au lieu de (ImageBlock, ImageBlock)
dblock_classif = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files,
    get_y=get_celeba_label,      # Extraction du label
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
    # Note : Si tu veux de la data augmentation légère pour la classification (ex: flip horizontal),
    # c'est ici qu'il faut l'ajouter via batch_tfms.
)

dls = dblock_classif.dataloaders(path_imgs, bs=BATCH_SIZE, num_workers=0)

# ==============================================================================
# 2. RECRÉER L'ARCHITECTURE ET CHARGER LE MODÈLE ENTRAÎNÉ
# ==============================================================================
print("Initialisation de l'architecture...")
model = AAE(
    input_size=256,
    input_channels=3, 
    encoding_dims=ENCODING_DIM,
    classes=2,        
)

# On recrée un Learner vide
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy])

# L'ÉTAPE MAGIQUE : On charge les poids du modèle que le callback a sauvegardé
# Fastai va chercher automatiquement le fichier 'brains_ae_classif_test.pth' dans le dossier 'models'
print("Injection des poids entraînés...")
learn.load(AAE_MODEL_NAME)



# ─────────────────────────────────────────────────────────────────────────────
#  2. EXTRACTION DE L'ESPACE LATENT (Z) À LA VOLÉE
# ─────────────────────────────────────────────────────────────────────────────
# ==============================================================================
# 4. EXTRACTION DE L'ESPACE LATENT ET DES LABELS SUR LE SET DE TEST
# ==============================================================================
# 4.1 Préparation du DataLoader de Test
print("Préparation du DataLoader de Test...")
items = get_image_files(path_imgs)
test_items = [item for item in items if part_dict.get(item.name) == 2]

# On crée un DataLoader de test rattaché au dls principal (pour hériter du vocabulaire)
test_dl = dls.test_dl(test_items, with_labels=True)

dev = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# 4.2 Extraction avec le callback GetLatentSpace
print("Extraction des vecteurs sur le set de test...")
learn.zi_valid = torch.tensor([]).to(dev)

# Au lieu d'utiliser ds_idx, on passe directement notre test_dl via l'argument 'dl'
_, all_targs = learn.get_preds(dl=test_dl, cbs=[GetLatentSpace()])

new_zi = learn.zi_valid.clone()

print(f"Extraction terminée. Shape de l'espace latent (Test) : {new_zi.shape}")

# Traduction des identifiants tensoriels en textes (Male/Female) via le vocabulaire
vocab = dls.vocab
labels_text = [vocab[t.item()] for t in all_targs]

# ─────────────────────────────────────────────────────────────────────────────
#  3. CALCUL D'ALIGNEMENT DES ATTRIBUTS (Pearson r²)
# ─────────────────────────────────────────────────────────────────────────────
print("▶  Calcul de l'alignement des attributs...")

def pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    r, _ = stats.pearsonr(x.ravel(), y.ravel())
    return float(r ** 2)

def signed_r(x: np.ndarray, y: np.ndarray) -> float:
    r, _ = stats.pearsonr(x.ravel(), y.ravel())
    return float(r)

alignment_celeba = {}

for name in ATTR_NAMES:
    r2_attr = pearson_r2(attributes_dict[name], target_score)
    sr_attr = signed_r(attributes_dict[name], target_score)
    
    # Stockage de l'importance avec son signe directionnel
    alignment_celeba[name] = np.sign(sr_attr) * r2_attr

# Tri par ordre d'importance absolue
importance_rank = sorted(ATTR_NAMES, key=lambda n: abs(alignment_celeba[n]), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  4. RÉGRESSION DE LA CIBLE DANS L'ESPACE LATENT
# ─────────────────────────────────────────────────────────────────────────────
print("▶  Régression de la direction cible dans l'espace latent...")

# On apprend la relation entre nos features latentes Z et le score de sortie
reg = HuberRegressor(max_iter=500)
reg.fit(Z_all, target_score)

target_direction_latent = reg.coef_                     

print("Terminé ! Le vecteur de direction latent a été calculé avec succès.")