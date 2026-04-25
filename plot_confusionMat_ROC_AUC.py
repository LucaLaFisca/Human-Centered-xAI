import torch
from fastai.vision.all import *
from fastai.data.all import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import datetime
import pandas as pd

# On importe ton architecture personnalisée
from modelAAE_DROPOUT import AAE
from utils import label_func # (et autres imports si nécessaires pour dls)

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
EPOCHS = 30
BATCH_SIZE = 16
ENCODING_DIM = 128
PATIENCE = 5
TARGET_ATTRIBUTE = 'Male' # L'attribut CelebA que tu souhaites classifier
AAE_MODEL_NAME = "CL_AAE_model_128" # Nom du modèle sauvegardé à charger

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

# ==============================================================================
# 3. INTERPRÉTATION ET GÉNÉRATION DES GRAPHIQUES
# ==============================================================================
print("Génération de la matrice de confusion...")

# Fastai fait passer le set de validation dans le modèle pré-entraîné
interp = ClassificationInterpretation.from_learner(learn)

# Création du dossier de résultats
out_dir = Path('results')
out_dir.mkdir(exist_ok=True)

interp.plot_confusion_matrix(figsize=(6, 6))

plt.title("Matrice de Confusion (Modèle Chargé)")
plt.savefig(out_dir / f'{AAE_MODEL_NAME}_confusion_matrix_loaded.png', dpi=300, bbox_inches='tight')

print(f"Succès ! Matrice sauvegardée dans : {out_dir / f'{AAE_MODEL_NAME}_confusion_matrix_loaded.png'}")

# ==============================================================================
# 4. GÉNÉRATION DE LA COURBE ROC ET DE L'AUC
# ==============================================================================
print("Génération de la courbe ROC AUC...")

# 1. Extraction des données depuis l'objet interp
# preds est un tenseur de dimension [N, 2] (probabilités pour Sain et Tumeur)
# targs est un tenseur de dimension [N] (vrais labels 0 ou 1)
preds, targs = learn.get_preds()

# 2. On isole les probabilités de la classe "positive" (Tumeur)
# Par convention dans Fastai, si tes dossiers sont ['Sain', 'Tumeur'], l'index 1 est Tumeur
y_score = preds[:, 1].numpy()
y_true = targs.numpy()

# 3. Calcul mathématique des taux et de l'aire avec Scikit-Learn
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# 4. Tracé du graphique
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.3f})')
# Ligne de la "chance" (modèle aléatoire)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 5. Sauvegarde
plt.savefig(out_dir / f'{AAE_MODEL_NAME}_roc_auc_loaded.png', dpi=300, bbox_inches='tight')
print(f"Succès ! Courbe ROC sauvegardée dans : {out_dir / f'{AAE_MODEL_NAME}_roc_auc_loaded.png'}")