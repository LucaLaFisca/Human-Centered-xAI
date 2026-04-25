import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from fastai.callback.tracker import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from fastai.callback.training import GradientAccumulation
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import pandas as pd

from modelAAE_DROPOUT import AAE
from utils import LossAttrMetric

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
EPOCHS = 50
BATCH = 16
ENCODING_DIM = 128
MASK_RATIO = 0.20  
PATCH_SIZE = 16    
LOSS_ALPHA = 0.84  
NOISE_STD = 0.05
PATIENCE = 10

SKIP_DROPOUT = 1

# Dossier d'output (on peut réutiliser le dossier de l'entraînement ou un nouveau)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"celeba_visu_{timestamp}"
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
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
    def encodes(self, x: TensorImage):
        x = x.clone()
        B, C, H, W = x.shape
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        n_masked = int((n_patches_h * n_patches_w) * self.mask_ratio)
        
        for b in range(B):
            indices = torch.randperm(n_patches_h * n_patches_w, device=x.device)[:n_masked]
            for idx in indices:
                i = (idx // n_patches_w) * self.patch_size
                j = (idx % n_patches_w) * self.patch_size
                x[b, :, i:i+self.patch_size, j:j+self.patch_size] = 0.
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
# 2. CHARGEMENT DES DONNÉES ET MODÈLE
# ==============================================================================
path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 
partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'
df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0: train_idx.append(i)
        elif part == 1: valid_idx.append(i)
    return train_idx, valid_idx

dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock), 
    get_items=get_image_files,
    get_y=lambda x: x,
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)

# Utilisation d'un device (GPU/CPU) pour l'inférence
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dls = dblock.dataloaders(path_imgs, bs=BATCH, num_workers=0)

model = AAE(
    input_size=256,
    input_channels=3, 
    encoding_dims=ENCODING_DIM,
    classes=2
).to(dev)

learn = Learner(dls, model, loss_func=AAEDenoisingLoss(), cbs=[corruption_cb])
learn.load('best_celeba_AE_with_LRfixed')
learn.model.eval()

# PREPARATION DU SET DE TEST (Partition 2)
items = get_image_files(path_imgs)
test_items = [item for item in items if part_dict.get(item.name) == 2]
test_dl = dls.test_dl(test_items, with_labels=True)

# ==============================================================================
# 3. FONCTION DE VISUALISATION
# ==============================================================================
def visualize_reconstruction(batch_data, dataset_name, save_filename, num_images=4):
    clean_xb, _ = batch_data
    clean_xb = clean_xb[:num_images].to(dev).clone()
    
    # Corruption manuelle pour la visualisation
    corrupted_xb = mask_tfm(noise_tfm(clean_xb.clone()))
    
    with torch.no_grad():
        # Forward pass
        _ = learn.model(corrupted_xb) 
        reconstructions = learn.model.decoder_output

    clean_img = clean_xb.cpu().clamp(0, 1)
    corr_img = corrupted_xb.cpu().clamp(0, 1)
    rec_img = reconstructions.cpu().clamp(0, 1)

    fig, axes = plt.subplots(3, num_images, figsize=(3 * num_images, 9))
    fig.suptitle(f"Reconstructions MS-SSIM - {dataset_name.upper()}", fontsize=16, fontweight='bold')
    
    for i in range(num_images):
        axes[0, i].imshow(clean_img[i].permute(1, 2, 0))
        if i == 0: axes[0, i].set_ylabel("1. Propre (Target)", size='large')
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        
        axes[1, i].imshow(corr_img[i].permute(1, 2, 0))
        if i == 0: axes[1, i].set_ylabel("2. Input Bruit/Masque", size='large')
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        
        axes[2, i].imshow(rec_img[i].permute(1, 2, 0))
        if i == 0: axes[2, i].set_ylabel("3. Sortie Décodeur", size='large')
        axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

    plt.tight_layout()
    full_path = OUT_DIR/save_filename
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close() # Libère la mémoire
    print(f"[{dataset_name}] Grille sauvegardée sous '{full_path}'")

# ==============================================================================
# 4. EXÉCUTION
# ==============================================================================
if __name__ == '__main__':
    print(f"Début de la visualisation. Résultats sauvegardés dans : {OUT_DIR}")
    
    #print("Génération des résultats sur le set d'ENTRAÎNEMENT (Train)...")
    #train_batch = dls.train.one_batch() 
    #visualize_reconstruction(train_batch, "Entraînement", "recons_train.png")

    #print("Génération des résultats sur le set de VALIDATION (Valid)...")
    #valid_batch = dls.valid.one_batch()
    #visualize_reconstruction(valid_batch, "Validation", "recons_valid.png")

    print("Génération des résultats sur le set de TEST (Test)...")
    test_batch = test_dl.one_batch()
    visualize_reconstruction(test_batch, "Test", "recons_test_celebA.png")
    
    print("Visualisation terminée.")