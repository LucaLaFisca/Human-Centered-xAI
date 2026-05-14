import torch
import torch.nn.functional as F
import pandas as pd
from fastai.vision.all import *
from fastai.callback.tracker import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from modelAAE_DROPOUT import AAE

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
EPOCHS = 30
BATCH_SIZE = 16
ENCODING_DIM = 128
PATIENCE = 5
TARGET_ATTRIBUTE = 'Male' # L'attribut CelebA que tu souhaites classifier

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"celeba_classifier_{timestamp}"
OUT_DIR = Path(f"results/{RUN_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 

# ==============================================================================
# 1. PRÉPARATION DES DONNÉES ET LABELS (CELEBA)
# ==============================================================================
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'
df_attr = pd.read_csv(attr_file, sep='\s+', header=1)

attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

def get_celeba_label(img_path):
    return attr_dict.get(img_path.name)

partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0: train_idx.append(i)
        elif part == 1: valid_idx.append(i)
    return train_idx, valid_idx

# ==============================================================================
# 2. INJECTION DU FILTRE ROUGE BIAISÉ ET SAUVEGARDE VISUELLE
# ==============================================================================

def get_biased_image(img_path):
    """Charge l'image et remplace le canal rouge selon le label."""
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    label = get_celeba_label(img_path) 
    h, w = img_np.shape[0], img_np.shape[1]
    
    # Génération du bruit aléatoire selon la classe
    if label == TARGET_ATTRIBUTE:  # Si c'est 'Male'
        noise = np.random.randint(128, 256, size=(h, w), dtype=np.uint8)
    else:                          # Si c'est 'Not Male'
        noise = np.random.randint(0, 128, size=(h, w), dtype=np.uint8)
        
    # Remplacement du canal rouge
    img_np[:, :, 0] = noise
    return PILImage.create(img_np)

# --- SCRIPT DE VÉRIFICATION (Sauvegarde en PNG au lieu d'afficher) ---
print("Génération de l'image de vérification du filtre...")
toutes_les_images = get_image_files(path_imgs)
images_test = random.sample(list(toutes_les_images), 4)

fig, axes = plt.subplots(4, 2, figsize=(8, 12))
fig.suptitle("Vérification du Filtre Rouge sur CelebA", fontsize=16)

for i, img_path in enumerate(images_test):
    img_orig = Image.open(img_path)
    label = get_celeba_label(img_path)
    img_biased = get_biased_image(img_path)
    
    axes[i, 0].imshow(img_orig)
    axes[i, 0].set_title(f"Originale ({label})")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(img_biased)
    axes[i, 1].set_title(f"Biaisée (Rouge {'Fort' if label==TARGET_ATTRIBUTE else 'Faible'})")
    axes[i, 1].axis('off')

plt.tight_layout()

# Sauvegarde de l'image dans le dossier de résultats du run courant
chemin_image_verif = OUT_DIR / "verification_filtre_rouge.png"
plt.savefig(chemin_image_verif, dpi=150, bbox_inches='tight')
plt.close() # Libération indispensable de la RAM sur le serveur
print(f"✅ Image de vérification sauvegardée sous : {chemin_image_verif}")
print("Lancement de la préparation des données...")
# -----------------------------------------------------------------------------

# --- Création du DataBlock avec le filtre (get_x) ---
dblock_classif = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files,
    get_x=get_biased_image,       
    get_y=get_celeba_label,      
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)

dls = dblock_classif.dataloaders(path_imgs, bs=BATCH_SIZE, num_workers=0)

# ==============================================================================
# 3. ENVELOPPE POUR LA FONCTION DE PERTE
# ==============================================================================
class AAEClassifLossWrapper:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, pred, target, **kwargs):
        return self.model.pure_classif_loss_func(pred, target, **kwargs)

# ==============================================================================
# 4. INITIALISATION DU MODÈLE ET ENTRAÎNEMENT
# ==============================================================================

model = AAE(
    input_size=256,
    input_channels=3,
    encoding_dims=ENCODING_DIM
)

learn = Learner(
    dls, 
    model,
    loss_func=AAEClassifLossWrapper(model),
    metrics=[accuracy] 
)

print("Recherche du Learning Rate optimal pour la classification...")
# On désactive le plot automatique pour éviter une erreur serveur
suggested_lrs = learn.lr_find(show_plot=False) 
lr_max = suggested_lrs.valley

print("Début de l'entraînement du classifieur...")
learn.fit_one_cycle(EPOCHS, lr_max=lr_max, cbs=[
    CSVLogger(fname=OUT_DIR/'history_classif.csv'),
    SaveModelCallback(monitor='valid_loss', fname='best_celeba_classifier'),
    EarlyStoppingCallback(monitor='valid_loss', patience=PATIENCE)
])