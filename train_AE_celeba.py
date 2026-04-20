import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from fastai.callback.tracker import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from pathlib import Path
import datetime
import matplotlib.pyplot as plt

# Import de ton architecture et de tes outils personnalisés
# Assure-toi que model.py accepte input_channels=3 dans __init__
from model import AAE
from utils import LossAttrMetric

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
EPOCHS = 50
ENCODING_DIM = 512
MASK_RATIO = 0.20  # Légèrement augmenté pour des visages
PATCH_SIZE = 16    # Patchs plus grands pour masquer des traits (yeux, nez)
LOSS_ALPHA = 0.84  
NOISE_STD = 0.05
PATIENCE = 10

# Dossier d'output
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"celeba_aae_{timestamp}"
OUT_DIR = Path(f"results/{RUN_NAME}")
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
        self.mask_ratio, self.patch_size = mask_ratio, patch_size
    def encodes(self, x: TensorImage):
        x = x.clone()
        H, W = x.shape[-2:]
        n_patches_h, n_patches_w = H // self.patch_size, W // self.patch_size
        n_masked = int((n_patches_h * n_patches_w) * self.mask_ratio)
        indices = torch.randperm(n_patches_h * n_patches_w)[:n_masked]
        for idx in indices:
            i, j = (idx // n_patches_w) * self.patch_size, (idx % n_patches_w) * self.patch_size
            if x.dim() == 4: x[:, :, i:i+self.patch_size, j:j+self.patch_size] = 0.
            else:            x[:, i:i+self.patch_size, j:j+self.patch_size] = 0.
        return x

class CorruptionCallback(Callback):
    def __init__(self, corruption_tfms): self.corruption_tfms = corruption_tfms
    def before_batch(self):
        self.learn.clean_xb = self.learn.xb[0].clone()
        corrupted = self.learn.xb[0].clone()
        for tfm in self.corruption_tfms: corrupted = tfm(corrupted)
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
path_imgs = Path('celeba_mini_clean/img_align_celeba') 

# La transformation exacte pour préserver l'alignement
# Fastai va redimensionner l'image sans la déformer, puis ajouter des bordures noires
# pour atteindre un carré parfait de 256x256.
align_resize = Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)

import pandas as pd
from fastai.vision.all import *

# 1. Charger le fichier de partition avec Pandas
# sep='\s+' gère les espaces multiples de CelebA
df_partition = pd.read_csv('celeba_mini_clean/list_eval_partition.txt', sep='\s+', header=None, names=['image_id', 'partition'])
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
dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock), 
    get_items=get_image_files,
    get_y=lambda x: x,
    splitter=celeba_splitter,  # <--- NOUVEAU SPLITTER
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)

dls = dblock.dataloaders(path_imgs, bs=128, num_workers=0)

# ==============================================================================
# 3. INITIALISATION DU MODÈLE ET ENTRAÎNEMENT
# ==============================================================================
model = AAE(
    input_size=256,
    input_channels=3,  # IMPORTANT: 3 pour CelebA (RGB)
    encoding_dims=ENCODING_DIM
)

learn = Learner(
    dls, model,
    loss_func=AAEDenoisingLoss(),
    metrics=[LossAttrMetric("recons_loss")],
    cbs=[corruption_cb]
)
corruption_cb.learn = learn

print("Entraînement en cours...")
print("Recherche du Learning Rate optimal...")
lr_max = 1e-3 # Valeur au milieu de la pente descendante observée dans lr_find()

learn.fit_one_cycle(EPOCHS, lr_max=lr_max, cbs=[
    CSVLogger(fname=OUT_DIR/'history.csv'),
    SaveModelCallback(fname='best_celeba_AE_with_LRfixed'),
    EarlyStoppingCallback(patience=PATIENCE)
])

# ==============================================================================
# 4. VISUALISATION DES RÉSULTATS
# ==============================================================================
learn.model.eval()
clean_xb, _ = dls.valid.one_batch()
with torch.no_grad():
    corrupted_xb = mask_tfm(noise_tfm(clean_xb.clone()))
    _ = learn.model(corrupted_xb.to(clean_xb.device))
    reconstructions = learn.model.decoder_output

# Plotting
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
for i in range(4):
    axes[0,i].imshow(clean_xb[i].permute(1,2,0).cpu())
    axes[1,i].imshow(corrupted_xb[i].permute(1,2,0).cpu())
    axes[2,i].imshow(reconstructions[i].permute(1,2,0).cpu().clamp(0,1))
plt.savefig(OUT_DIR/'reconstructions_celeba.png')