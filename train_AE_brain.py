import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from fastai.callback.tracker import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from pathlib import Path
import datetime
import matplotlib.pyplot as plt

from model import AAE
from utils import LossAttrMetric

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
EPOCHS = 100
MASK_RATIO = 0.15
PATCH_SIZE = 8
LOSS_ALPHA = 0.84 # Ratio MS-SSIM vs L1
NOISE_STD = 0.05
PATIENCE = 15

# Création dynamique du dossier d'output avec horodatage
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"run_{timestamp}_ep{EPOCHS}_mask{MASK_RATIO}_alpha{LOSS_ALPHA}"
OUT_DIR = Path(f"results/{RUN_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Les résultats de cette session seront sauvegardés dans : {OUT_DIR}")

# ==============================================================================
# 1. DÉFINITION DES TRANSFORMATIONS (CORRUPTIONS)
# ==============================================================================
class AddGaussianNoise(Transform):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    
    def encodes(self, x: TensorImage):
        noise = torch.randn_like(x) * self.std + self.mean
        return (x + noise).clamp(0, 1)

class RandomMasking(Transform):
    def __init__(self, mask_ratio=0.3, patch_size=16):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
    
    def encodes(self, x: TensorImage):
        x = x.clone()
        H, W = x.shape[-2:] 
        
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        n_patches = n_patches_h * n_patches_w
        n_masked = int(n_patches * self.mask_ratio)
        
        indices = torch.randperm(n_patches)[:n_masked]
        
        for idx in indices:
            i = (idx // n_patches_w) * self.patch_size
            j = (idx % n_patches_w) * self.patch_size
            
            if x.dim() == 4: 
                x[:, :, i:i+self.patch_size, j:j+self.patch_size] = 0.
            else:            
                x[:, i:i+self.patch_size, j:j+self.patch_size] = 0.
        
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
    def __call__(self, pred, *yb): # Capture tous les arguments cibles sous forme de tuple
        learn = corruption_cb.learn
        clean = learn.clean_xb
        
        # Le *yb redéballe le tuple pour l'injecter proprement dans ta fonction
        return model.denoising_ae_loss_func(clean, pred, *yb)

noise_tfm = AddGaussianNoise(mean=0., std=NOISE_STD)
mask_tfm = RandomMasking(mask_ratio=MASK_RATIO, patch_size=PATCH_SIZE)

corruption_cb = CorruptionCallback(
    corruption_tfms=[noise_tfm, mask_tfm]
)


# ==============================================================================
# 3. CHARGEMENT DES DONNÉES (1 CANAL - NIVEAUX DE GRIS)
# ==============================================================================
data_path = Path('db_brain_tumor') 

dblock = DataBlock(
    # Forcer le chargement en 1 seul canal (Noir et Blanc)
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock()),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)

dls = dblock.dataloaders(data_path, bs=16, drop_last=True, num_workers=0)

# ==============================================================================
# 4. INITIALISATION DU MODÈLE ET ENTRAÎNEMENT
# ==============================================================================
model = AAE(
    input_size=256,
    input_channels=1, # Correspond au PILImageBW
    encoding_dims=1024,
    classes=2,        
)

metrics = [LossAttrMetric("recons_loss")]

learn = Learner(
    dls, model,
    loss_func=AAEDenoisingLoss(),
    metrics=metrics,
    cbs=[corruption_cb]
)

corruption_cb.learn = learn

# Définition du chemin de sauvegarde du modèle dans le dossier dynamique
model_file = OUT_DIR / '1024_ae_denoising_msssim_brains.pth'

print("Recherche du Learning Rate optimal...")
learning_rate = learn.lr_find(suggest_funcs=(valley,))
lr_max = learning_rate.valley

print(f"Début de l'entraînement avec lr_max={lr_max:.4f} (1cycle policy)...")
learn.fit_one_cycle(
    n_epoch=EPOCHS, 
    lr_max=lr_max,
    cbs=[
        CSVLogger(fname=OUT_DIR / 'history.csv'),
        SaveModelCallback(fname=model_file.name), # Fastai sauvegarde dans dls.path/models/ par défaut
        EarlyStoppingCallback(min_delta=1e-4, patience=PATIENCE)
    ]
)

# Sauvegarde du graphique d'entraînement
learn.recorder.plot_loss()
plt.savefig(OUT_DIR / 'loss_curve.png', bbox_inches='tight')
plt.close()

# ==============================================================================
# 5. CHARGEMENT ET GÉNÉRATION DE LA VISUALISATION
# ==============================================================================
print("Génération des visualisations sur le set de validation...")
learn.load(model_file.name) 
learn.model.eval()

clean_xb, yb = dls.valid.one_batch()

corrupted_xb = clean_xb.clone()
corrupted_xb = noise_tfm(corrupted_xb)
corrupted_xb = mask_tfm(corrupted_xb)

with torch.no_grad():
    learn.model.to(corrupted_xb.device)
    _ = learn.model(corrupted_xb) 
    reconstructions = learn.model.decoder_output

clean_img = clean_xb.cpu().clamp(0, 1)
corr_img = corrupted_xb.cpu().clamp(0, 1)
rec_img = reconstructions.cpu().clamp(0, 1)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
title_str = f"Reconstructions\nEpochs: {EPOCHS} | Mask: {MASK_RATIO} | LR: {lr_max:.1e} | Loss Alpha: {LOSS_ALPHA}"
fig.suptitle(title_str, fontsize=14, fontweight='bold')

for i in range(4):
    # .squeeze(0) retire la dimension de canal qui est maintenant de 1
    axes[0, i].imshow(clean_img[i].squeeze(0), cmap='gray')
    if i == 0: axes[0, i].set_ylabel("1. Propre (Target)", size='large')
    axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
    
    axes[1, i].imshow(corr_img[i].squeeze(0), cmap='gray')
    if i == 0: axes[1, i].set_ylabel("2. Input Bruit/Masque", size='large')
    axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    
    axes[2, i].imshow(rec_img[i].squeeze(0), cmap='gray')
    if i == 0: axes[2, i].set_ylabel("3. Sortie Décodeur", size='large')
    axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

plt.tight_layout()
recons_file = OUT_DIR / 'reconstructions.png'
plt.savefig(recons_file, bbox_inches='tight', dpi=300)
print(f"Succès ! Tous les résultats ont été sauvegardés dans : {OUT_DIR}")