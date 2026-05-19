import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import pandas as pd

from modelAAE_DROPOUT import AAE

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BATCH = 16
ENCODING_DIM = 128
MASK_RATIO = 0.20  
PATCH_SIZE = 16    
NOISE_STD = 0.05
TARGET_ATTRIBUTE = 'Male'
# Dossier d'output
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"celeba_visu_AAE_{timestamp}"
OUT_DIR = Path(f"CL_results/{RUN_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. TRANSFORMATIONS DE CORRUPTION 
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

noise_tfm = AddGaussianNoise(std=NOISE_STD)
mask_tfm = RandomMasking(mask_ratio=MASK_RATIO, patch_size=PATCH_SIZE)

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES ET DU MODÈLE AAE
# ==============================================================================
# Dataset preprocessed
# path_imgs = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/img_align_celeba')
# attr_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_attr_celeba.txt'
# partition_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_eval_partition.txt'
# dataset original
path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 
partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'



df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

# attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'
df_attr = pd.read_csv(attr_file, sep='\s+', header=1)

attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}
def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0: train_idx.append(i)   
        elif part == 1: valid_idx.append(i)
    return train_idx, valid_idx


# Fonction pour extraire le label (y) d'une image (x)
def get_celeba_label(img_path):
    return attr_dict.get(img_path.name)



dblock = DataBlock(
    blocks=(ImageBlock, ImageBlock), 
    get_items=get_image_files,
    # get_x=get_biased_image,
    get_y=lambda x: x,
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dls = dblock.dataloaders(path_imgs, bs=BATCH, num_workers=0)

# Instanciation du modèle avec les mêmes paramètres que dans main.py (classes=2)
model = AAE(
    input_size=256,
    input_channels=3, 
    encoding_dims=ENCODING_DIM,
    classes=2
).to(dev)

# Chargement direct des poids via PyTorch (strict=False pour ignorer les poids manquants/inutiles)
# weights_path = 'models/CL_AE_SOLO_model.pth'

weights_path = f'models/CL_CLASSIF_model_{ENCODING_DIM}.pth'
print(f"Chargement des poids depuis : {weights_path}")
state_dict = torch.load(weights_path, map_location=dev)
model.load_state_dict(state_dict, strict=False)

# Passage du modèle en mode évaluation (Désactive le Dropout et fige les BatchNorm)
model.eval()

# PREPARATION DU SET DE TEST (Partition 2)
items = get_image_files(path_imgs)
test_items = [item for item in items if part_dict.get(item.name) == 2]
test_dl = dls.test_dl(test_items, with_labels=True)

# ==============================================================================
# 3. FONCTION DE VISUALISATION (INFERENCE DIRECTE)
# ==============================================================================
def visualize_reconstruction(batch_data, dataset_name, save_filename, num_images=4):
    clean_xb, _ = batch_data
    clean_xb = clean_xb[:num_images].to(dev).clone()
    
    # Corruption manuelle pour tester la robustesse
    corrupted_xb = mask_tfm(noise_tfm(clean_xb.clone()))
    
    with torch.no_grad():
        # 1. Forward pass sur l'image corrompue (Test Denoising)
        _ = model(corrupted_xb) 
        reconstructions_corr = model.decoder_output.clone()
        
        # 2. Forward pass sur l'image propre (Test Autoencodeur pur pour xAI)
        _ = model(clean_xb)
        reconstructions_clean = model.decoder_output.clone()

    clean_img = clean_xb.cpu().clamp(0, 1)
    corr_img = corrupted_xb.cpu().clamp(0, 1)
    rec_corr_img = reconstructions_corr.cpu().clamp(0, 1)
    rec_clean_img = reconstructions_clean.cpu().clamp(0, 1)

    # Création d'une grille à 4 lignes
    fig, axes = plt.subplots(4, num_images, figsize=(3 * num_images, 12))
    fig.suptitle(f"Reconstructions AAE - {dataset_name.upper()}", fontsize=16, fontweight='bold')
    
    for i in range(num_images):
        # Ligne 1 : Image propre
        axes[0, i].imshow(clean_img[i].permute(1, 2, 0))
        if i == 0: axes[0, i].set_ylabel("1. Propre (Target)", size='large')
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        
        # Ligne 2 : Image corrompue
        axes[1, i].imshow(corr_img[i].permute(1, 2, 0))
        if i == 0: axes[1, i].set_ylabel("2. Input Bruit/Masque", size='large')
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        
        # Ligne 3 : Reconstruction depuis l'image corrompue
        axes[2, i].imshow(rec_corr_img[i].permute(1, 2, 0))
        if i == 0: axes[2, i].set_ylabel("3. Sortie (Denoising)", size='large')
        axes[2, i].set_xticks([]); axes[2, i].set_yticks([])
        
        # Ligne 4 : Reconstruction depuis l'image propre
        axes[3, i].imshow(rec_clean_img[i].permute(1, 2, 0))
        if i == 0: axes[3, i].set_ylabel("4. Sortie (AE Pur)", size='large')
        axes[3, i].set_xticks([]); axes[3, i].set_yticks([])

    plt.tight_layout()
    full_path = OUT_DIR/save_filename
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[{dataset_name}] Grille sauvegardée sous '{full_path}'")

# ==============================================================================
# 4. EXÉCUTION
# ==============================================================================
if __name__ == '__main__':
    print(f"Début de la visualisation. Résultats sauvegardés dans : {OUT_DIR}")
    
    print("Génération des résultats sur le set de TEST (Test)...")
    test_batch = test_dl.one_batch()
    visualize_reconstruction(test_batch, "Test", "recons_test_aae_celebA.png")
    
    print("Visualisation terminée.")