import torch
from fastai.vision.all import *
from fastai.data.all import *
from fastai.callback.tracker import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from fastai.callback.training import GradientAccumulation
from pathlib import Path
import pandas as pd

from modelAAE_DROPOUT import AAE
from utils import UnfreezeFcCritAdaptative, label_func, FreezeDiscriminator, GetLatentSpace, LossAttrMetric, distrib_regul_regression, compute_main_direction

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns



# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
EPOCHS_AE = 50
EPOCHS_CLASSIF = 30
EPOCHS_ADV = 30

LR_MAX_FACTOR = 5 # par exemple 

BATCH = 16
ENCODING_DIM = 128
MASK_RATIO = 0.20  # Légèrement augmenté pour des visages
PATCH_SIZE = 16    # Patchs plus grands pour masquer des traits (yeux, nez)
LOSS_ALPHA = 0.84  
NOISE_STD = 0.05
PATIENCE = 10

# POIDS DES LOSS
class_RECONS_WEIGHT = 0.1
class_CLASS_WEIGHT = 0.9
adv_RECONS_WEIGHT = 0.04
adv_CLASS_WEIGHT = 0.01
adv_ADV_WEIGHT = 0.95


# Param de l'ADVERSARIAL
LOW_TESH = 0.65
HIGH_TESH = 0.80

# Dossier d'output

RUN_NAME = (
    f"CL_celeba_aae_"
    f"lr{str(LR_MAX_FACTOR).replace('.', 'p')}_"
    f"enc{ENCODING_DIM}_"
    f"low{str(LOW_TESH).replace('.', 'p')}_high{str(HIGH_TESH).replace('.', 'p')}_"
    f"classpoids{str(class_RECONS_WEIGHT).replace('.', 'p')}-wc{str(class_CLASS_WEIGHT).replace('.', 'p')}_"
    f"advpoids{str(adv_RECONS_WEIGHT).replace('.', 'p')}-ac{str(adv_CLASS_WEIGHT).replace('.', 'p')}-aa{str(adv_ADV_WEIGHT).replace('.', 'p')}_"
)
OUT_DIR = Path(f"CL_results/{RUN_NAME}")
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
# 2. CHARGEMENT DES DONNÉES CELEBA (RGB)
# ==============================================================================
# Remplace par ton chemin vers le dossier img_align_celeba dézippé
path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 

partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
# 1. Charger le fichier de partition avec Pandas
# sep='\s+' gère les espaces multiples de CelebA
df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])
# On le transforme en dictionnaire pour que Fastai le lise instantanément
# Format attendu : {'000001.jpg': 0, '000002.jpg': 1, ...}
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

# 2. Créer le Splitter sur mesure
def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0:
            train_idx.append(i)  # 0 : Training
        elif part == 1:
            valid_idx.append(i)  # 1 : Validation
        # Si part == 2 (Test), on l'ignore silencieusement. 
        # Il ne polluera ni le train ni le valid de l'AE.
    return train_idx, valid_idx

# 3. L'intégrer dans ton DataBlock actuel
# La transformation exacte pour préserver l'alignement
# Fastai va redimensionner l'image sans la déformer, puis ajouter des bordures noires
# pour atteindre un carré parfait de 256x256.
align_resize = Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock), 
    get_items=get_image_files,
    get_y=lambda x: x,
    splitter=celeba_splitter,  # <--- NOUVEAU SPLITTER
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

#==============================================================================
# 4. ENTRAINEMENT DE L'AUTOENCODER
#==============================================================================

learn = Learner(
    dls, model,
    loss_func=AAEDenoisingLoss(),
    metrics=[LossAttrMetric("recons_loss")],
    cbs=[corruption_cb]
)
corruption_cb.learn = learn

print("Entraînement en cours...")
print("Recherche du Learning Rate optimal...")
lr_max = learn.lr_find().valley # Valeur au milieu de la pente descendante observée dans lr_find()

model_file = 'CL_AE_model'
learn.fit_one_cycle(EPOCHS_AE, lr_max=lr_max,
            cbs=[TrackerCallback(),
                 SaveModelCallback(fname=model_file),
                 EarlyStoppingCallback(min_delta=1e-4,patience=10),
                 GradientAccumulation(n_acc=4)])

state_dict = torch.load(f'models/{model_file}.pth')
model.load_state_dict(state_dict, strict=False)

#==============================================================================
# 5. ENTRAINEMENT DU CLASSIFIEUR (en gardant les poids de l'AE)
#==============================================================================

metrics = [#LossAttrMetric("adv_loss"), 
           LossAttrMetric("recons_loss"),
           LossAttrMetric("classif_loss"),
           #LossAttrMetric("crit_loss"),
           accuracy] # formule de accuracy = nombre de prédictions correctes / nombre total d'exemples
monitor_loss = 'valid_loss'
learn = Learner(dls, model, loss_func=model.pure_classif_loss_func(RECONS_WEIGHT=class_RECONS_WEIGHT, CLASS_WEIGHT=class_CLASS_WEIGHT), metrics=metrics)

model_file = 'CL_CLASSIF_model'
learn.fit(EPOCHS_CLASSIF, lr=lr_max/LR_MAX_FACTOR,
            cbs=[GradientAccumulation(n_acc=4),
                 TrackerCallback(monitor=monitor_loss),
                 SaveModelCallback(fname=model_file,monitor=monitor_loss),
                 EarlyStoppingCallback(min_delta=1e-4,patience=10,monitor=monitor_loss),
               #   FreezeDiscriminator()])
                ])

lr= lr_max/LR_MAX_FACTOR

#==============================================================================
# 6. ENTRAINEMENT ADVERSARIAL 
#==============================================================================
metrics = [LossAttrMetric("adv_loss"),
           LossAttrMetric("recons_loss"),
           LossAttrMetric("classif_loss"),
           accuracy
           ]
learn = Learner(dls, model, loss_func=model.aae_loss_func(RECONS_WEIGHT=adv_RECONS_WEIGHT, CLASS_WEIGHT=adv_CLASS_WEIGHT, ADV_WEIGHT=adv_ADV_WEIGHT), metrics=metrics)

model_file = 'CL_AAE_model'
learn.fit(EPOCHS_ADV, lr=lr/LR_MAX_FACTOR,
            cbs=[GradientAccumulation(n_acc=4),
                 TrackerCallback(),
                 SaveModelCallback(fname=model_file),
                 EarlyStoppingCallback(min_delta=1e-4,patience=10),
               #   FreezeDiscriminator()]),
                 UnfreezeFcCritAdaptative(low_threshold=LOW_TESH, high_threshold=HIGH_TESH)])

state_dict = torch.load(f'models/{model_file}.pth')
model.load_state_dict(state_dict, strict=False)


#==============================================================================
# 7. EXTRACTION DE ZE, VISUALISATION, ET SAUVEGARDE FINALE
#==============================================================================
#learn.load(f'models/{model_file}', strict=False)
learn.load(model_file, strict=False)

torch.save(model.state_dict(), f'models/{model_file}_{ENCODING_DIM}.pth')
print(f"PTH sauvegardé : models/{model_file}_{ENCODING_DIM}.pth")

# ── Extraire Ze ──────────────────────────────────────────────────────
dev = f'cuda:{torch.cuda.current_device()}'
learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=0, cbs=[GetLatentSpace()])
new_zi = learn.zi_valid.clone()

learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=1, cbs=[GetLatentSpace()])
new_zi = torch.vstack((new_zi, learn.zi_valid))

torch.save(new_zi, f'z_{ENCODING_DIM}.pt')
print(f"Ze shape : {new_zi.shape}")

# ── Labels depuis le DataLoader (alignés sur new_zi via drop_last) ───
#train_labels = torch.cat([y for _, y in dls.train], dim=0)
#valid_labels = torch.cat([y for _, y in dls.valid], dim=0)
#lab_gather   = torch.cat([train_labels, valid_labels], dim=0)

#N_min      = min(len(lab_gather), len(new_zi))
#lab_gather = lab_gather[:N_min, 1].float().cpu()  # 0.0=cat, 1.0=dog
#category   = ['dog' if l == 1 else 'cat' for l in lab_gather.numpy()]