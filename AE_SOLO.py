import torch
from fastai.vision.all import *
from fastai.data.all import *
from fastai.callback.tracker import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from fastai.callback.training import GradientAccumulation
from pathlib import Path
import pandas as pd

from modelAAE_DROPOUT import AAE
from utils_AAE import UnfreezeFcCritAdaptative, label_func, FreezeDiscriminator, GetLatentSpace, LossAttrMetric, distrib_regul_regression, compute_main_direction

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
TARGET_ATTRIBUTE = 'Male' 

EPOCHS_AE = 50
EPOCHS_CLASSIF = 30
EPOCHS_ADV = 35

LR_MAX_FACTOR = 3 # par exemple 

BATCH = 16
ENCODING_DIM = 128
MASK_RATIO = 0.20  # Légèrement augmenté pour des visages
PATCH_SIZE = 16    # Patchs plus grands pour masquer des traits (yeux, nez)
LOSS_ALPHA = 0.84  
NOISE_STD = 0.05
PATIENCE = 10

# POIDS DES LOSS (Manquants dans ton script d'origine)
ae_RECONS_WEIGHT = 0.9
ae_ADV_WEIGHT = 0.1
class_RECONS_WEIGHT = 0.79
class_CLASS_WEIGHT = 0.001
class_ADV_WEIGHT = 0.2

# Learning rate manquant
lr_max = 1e-3
# Param de l'ADVERSARIAL
LOW_TESH = 0.65
HIGH_TESH = 0.80

# Dossier d'output
RUN_NAME = (
    f"CL_celeba_aae_"
    f"lr{str(LR_MAX_FACTOR).replace('.', 'p')}_"
    f"enc{ENCODING_DIM}_"
    f"low{str(LOW_TESH).replace('.', 'p')}_high{str(HIGH_TESH).replace('.', 'p')}_"
    f"classpoids{str(ae_RECONS_WEIGHT).replace('.', 'p')}-wc{str(ae_ADV_WEIGHT).replace('.', 'p')}_"
    f"advpoids{str(class_RECONS_WEIGHT).replace('.', 'p')}-ac{str(class_CLASS_WEIGHT).replace('.', 'p')}-aa{str(class_ADV_WEIGHT).replace('.', 'p')}_"
)
OUT_DIR = Path(f"AE_results/{RUN_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. TRANSFORMATIONS ET CALLBACK DE CORRUPTION 
# ==============================================================================
class AddGaussianNoise(Transform):
    def __init__(self, mean=0., std=0.05):
        self.mean, self.std = mean, std
    def encodes(self, x: TensorImage):
        noise = torch.randn_like(x) * self.std + self.mean
        return (x + noise).clamp(0, 1)

class RandomMasking(Transform):
    def __init__(self, mask_ratio=0.3, patch_size=16):
        # Initialisation des hyperparamètres de masquage
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
    def encodes(self, x: TensorImage):
        # On clone le tenseur entrant pour ne pas altérer la donnée d'origine en mémoire.
        x = x.clone()

        B, C, H, W = x.shape
        
        # Calcul du nombre total de patchs possibles sur la hauteur (H) et la largeur (W).
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        
        # Calcul du nombre absolu de patchs à masquer pour cette image.
        # On multiplie le nombre total de patchs par le ratio désiré, et on force la conversion en entier.
        n_masked = int((n_patches_h * n_patches_w) * self.mask_ratio)
        
        # Boucle itérant sur chaque image individuelle du batch
        for b in range(B):
            # Génération d'une séquence de nombres aléatoires sans remise.
            # On génère un index 1D pour chaque patch possible, on les mélange (randperm), 
            # et on garde seulement les 'n_masked' premiers éléments.
            # L'argument 'device=x.device' garantit que cette opération se fait sur le GPU si 'x' y est déjà.
            indices = torch.randperm(n_patches_h * n_patches_w, device=x.device)[:n_masked]
            
            # Application du masque noir (0.) pour chaque index aléatoire tiré
            for idx in indices:
                # --- CONVERSION DE L'INDEX 1D EN COORDONNÉES 2D (LIGNE / COLONNE) ---
                
                # Coordonnée 'i' (axe Y, hauteur) :
                # La division entière '//' par la largeur de la grille donne le numéro de la ligne (la rangée).
                # Ex: Si la grille fait 16 patchs de large, l'index 34 correspond à la rangée 2 (34 // 16 = 2).
                # On multiplie ensuite par 'patch_size' pour obtenir le pixel de départ réel sur l'image.
                i = (idx // n_patches_w) * self.patch_size
                
                # Coordonnée 'j' (axe X, largeur) :
                # Le modulo '%' par la largeur de la grille donne le reste, c'est-à-dire la position dans la ligne (la colonne).
                # Ex: Pour l'index 34 sur une largeur de 16, le reste est 2 (34 % 16 = 2). C'est la 3ème colonne.
                # On multiplie par 'patch_size' pour obtenir le pixel de départ horizontal.
                j = (idx % n_patches_w) * self.patch_size
                
                # --- APPLICATION DU MASQUE ---
                # b : on cible l'image courante de la boucle
                # : : on cible tous les canaux de couleur (R, G, B) en même temps
                # i:i+self.patch_size : on cible les pixels en hauteur sur la taille du patch
                # j:j+self.patch_size : on cible les pixels en largeur sur la taille du patch
                # On met toutes ces valeurs à 0. (noir)
                x[b, :, i:i+self.patch_size, j:j+self.patch_size] = 0.
                
        # On retourne le batch complet, désormais masqué de manière indépendante pour chaque image.
        return x

class CorruptionCallback(Callback):
    def __init__(self, corruption_tfms): 
        self.corruption_tfms = corruption_tfms
    def before_batch(self):
        self.learn.clean_xb = self.learn.xb[0].clone()
        corrupted = self.learn.xb[0].clone()
        for tfm in self.corruption_tfms: 
            corrupted = tfm(corrupted)
        self.learn.xb = (corrupted,)

class AAEDenoisingLoss:
    def __call__(self, pred, *yb):
        clean = corruption_cb.learn.clean_xb
        return model.denoising_ae_loss_func(clean, pred, *yb)

noise_tfm = AddGaussianNoise(std=NOISE_STD)
mask_tfm = RandomMasking(mask_ratio=MASK_RATIO, patch_size=PATCH_SIZE)
corruption_cb = CorruptionCallback(corruption_tfms=[noise_tfm, mask_tfm])

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES CELEBA (RGB) ET LABELS
# ==============================================================================
# Dataset preprocessed
path_imgs = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/img_align_celeba')
attr_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_attr_celeba.txt'
partition_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_eval_partition.txt'

# --- A. Chargement de la partition ---
df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

# --- B. Chargement de l'attribut cible ---

df_attr = pd.read_csv(attr_file, sep='\s+', header=1)

attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
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

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,

    get_y=get_celeba_label,        # retourne la string "Male" / "Not Male"
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)
dls = dblock.dataloaders(path_imgs, bs=BATCH, num_workers=0)

# ==============================================================================
# 3. INITIALISATION DU MODÈLE ET ENTRAÎNEMENT
# ==============================================================================
model = AAE(
    input_size=256,
    input_channels=3,  # R G B 
    encoding_dims=ENCODING_DIM
)
# ON CHANGE L'ordre avec adversarial en premier
#==============================================================================
# 4. ENTRAINEMENT AE (SOLO)
#==============================================================================
print("Entraînement AE Solo...")

# 1. Gel strict des paramètres du Discriminateur et du Classifieur
for param in model.latent_gan.parameters():
    param.requires_grad = False

for param in model.linear.parameters():
    param.requires_grad = False

# 2. S'assurer que les gradients sont activés pour Encodeur et Décodeur
for param in model.encoder.parameters():
    param.requires_grad = True
for param in model.fc_encode.parameters():
    param.requires_grad = True
for param in model.decoder_fc.parameters():
    param.requires_grad = True
for param in model.unet.parameters():
    param.requires_grad = True

# 3. Définition du wrapper de Loss adapté à ta nouvelle fonction solo
class AELoss:
    def __init__(self):
        # Plus besoin de recons_weight et class_weight ici !
        pass

    def __call__(self, pred, *yb):
        # Récupération de l'image d'origine sans bruit via le callback
        clean_xb = corruption_cb.learn.clean_xb  
        
        # Appel de ta fonction spécifique pour l'AE Solo
        return model.ae_loss_func_solo(clean_xb)
                 
ae_loss = AELoss() 

learn = Learner(
    dls, model,
    loss_func=ae_loss,
    # Utilise l'attribut self.recons_loss stocké dans ae_loss_func_solo
    metrics=[LossAttrMetric("recons_loss")], 
    cbs=[corruption_cb]
)
corruption_cb.learn = learn

model_file = 'CL_AE_SOLO_model'
print(f"Entraînement de l'autoencodeur avec lr_max={(lr_max/LR_MAX_FACTOR):.2e}...")

# 4. Lancement de l'entraînement
learn.fit_one_cycle(EPOCHS_AE, lr_max=lr_max/LR_MAX_FACTOR,
            cbs=[TrackerCallback(monitor='valid_loss'),
                 SaveModelCallback(fname=model_file, monitor='valid_loss'),
                 EarlyStoppingCallback(min_delta=1e-4, patience=PATIENCE, monitor='valid_loss'),
                 GradientAccumulation(n_acc=4)])

# 5. Chargement des poids (strict=False est crucial car le discriminateur était gelé)
state_dict = torch.load(f'models/{model_file}.pth')
model.load_state_dict(state_dict, strict=False)
print("Entraînement AE Solo terminé avec succès.")