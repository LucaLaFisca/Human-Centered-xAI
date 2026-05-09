import torch
import torch.nn.functional as F
from fastai.vision.all import *
from pytorch_msssim import ms_ssim

from model import AAE
from utils import UnfreezeFcCritAdaptative, label_func, FreezeDiscriminator, GetLatentSpace, LossAttrMetric, distrib_regul_regression, compute_main_direction

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

class AddGaussianNoise(Transform):
    """Ajoute du bruit gaussien à l'image pour le denoising"""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    
    def encodes(self, x: TensorImage):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

class RandomMasking(Transform):
    """Masque aléatoirement des patches de l'image (MAE-style)"""
    def __init__(self, mask_ratio=0.3, patch_size=16):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
    
    def encodes(self, x: TensorImage):
        x = x.clone()
        
        # Récupère H et W depuis la fin du tuple, peu importe si c'est 3D ou 4D
        H, W = x.shape[-2:] 
        
        # Nombre de patches
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        n_patches = n_patches_h * n_patches_w
        n_masked = int(n_patches * self.mask_ratio)
        
        # Sélection aléatoire des patches à masquer
        indices = torch.randperm(n_patches)[:n_masked]
        
        for idx in indices:
            i = (idx // n_patches_w) * self.patch_size
            j = (idx % n_patches_w) * self.patch_size
            
            # Gestion de la dimension du batch
            if x.dim() == 4: # Si c'est un batch : [Batch, Channel, H, W]
                x[:, :, i:i+self.patch_size, j:j+self.patch_size] = 0.
            else:            # Si c'est une image seule : [Channel, H, W]
                x[:, i:i+self.patch_size, j:j+self.patch_size] = 0.
        
        return x
    
class CorruptionCallback(Callback):
    """
    Corrompt les inputs avant le forward pass.
    Le target de reconstruction devient l'image originale propre.
    """
    def __init__(self, corruption_tfms):
        self.corruption_tfms = corruption_tfms  # liste de transforms
    
    def before_batch(self):
        # Sauvegarde l'image propre comme target de reconstruction
        self.learn.clean_xb = self.learn.xb[0].clone()
        
        # Applique les corruptions sur l'input
        corrupted = self.learn.xb[0].clone()
        for tfm in self.corruption_tfms:
            corrupted = tfm(corrupted)
        
        # Remplace le batch input par la version corrompue
        self.learn.xb = (corrupted,)

class CleanTargetLossWrapper(Module):
    """
    Wrapper qui injecte clean_xb dans la loss à la place de xb corrompu.
    """
    def __init__(self, model, base_loss_func):
        self.model = model
        self.base_loss_func = base_loss_func
    
    def forward(self, pred, *yb):
        return self.base_loss_func(pred, *yb)
    
    def __call__(self, pred, *yb):
        # Récupère le clean_xb stocké par le callback
        if hasattr(self, '_learn') and hasattr(self._learn, 'clean_xb'):
            return self.base_loss_func(self._learn.clean_xb, pred, *yb)
        return self.base_loss_func(pred, *yb)
    
# --- Dans la partie entraînement ---
# Définition des corruptions
noise_tfm  = AddGaussianNoise(mean=0., std=0.05)
light_mask_tfm = RandomMasking(mask_ratio=0.1, patch_size=32)
mask_tfm   = RandomMasking(mask_ratio=0.3, patch_size=16)

# Callback avec les deux corruptions combinées
corruption_cb = CorruptionCallback(
    corruption_tfms=[noise_tfm]  # appliquées séquentiellement
)

# Loss adaptée pour utiliser clean_xb
class AAEDenoisingLoss:
    def __call__(self, pred, *yb):
        learn = corruption_cb.learn  # accès via le callback
        clean = learn.clean_xb
        return model.denoising_ae_loss_func(clean, pred, *yb)

### Define the Dataloader
data_path = untar_data(URLs.PETS) #checker les autres databases dispo

print(data_path.ls())
catblock = MultiCategoryBlock(encoded=True, vocab=['cat', 'dog'])
# DataBlock (batch_tfms sans corruption — la corruption est dans le callback)
dblock = DataBlock(
    blocks=(ImageBlock(), catblock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_func,
    item_tfms=Resize(256, method="squish"),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],  # pas de corruption ici
)

dls = dblock.dataloaders(data_path/"images", bs=16, drop_last=True, num_workers=0)

model = AAE(
    input_size=256,
    input_channels=3,
    encoding_dims=512,
    classes=2,
)

metrics = [LossAttrMetric("recons_loss")]
learn = Learner(
    dls, model,
    loss_func=AAEDenoisingLoss(),
    metrics=metrics,
    cbs=[corruption_cb]         # <-- injection du callback
)

# Lier le callback au learn pour accès à clean_xb dans la loss
corruption_cb.learn = learn

model_file = 'cat_dog_ae_denoising'
learning_rate = learn.lr_find()
learn.fit(
    100, lr=learning_rate.valley,
    cbs=[
        TrackerCallback(),
        SaveModelCallback(fname=model_file),
        EarlyStoppingCallback(min_delta=1e-4, patience=10)
    ]
)

print("\nEntraînement terminé, modèle sauvegardé sous:", model_file)
print("learning_rate trouvé:", learning_rate.valley)
print("C'est arrêté à l'époque:", learn.epoch)
print("Meilleure métrique enregistrée:", learn.recorder.metrics[-1])
print("Meilleure perte enregistrée:", learn.recorder.losses[-1])
print("Meilleur score de validation enregistré:", learn.recorder.values[-1][0])
print("Meilleur score de validation enregistré (recons_loss):", learn.recorder.values[-1][1])


# --- 5. CHARGER LES POIDS ET GÉNÉRER LES IMAGES ---
learn.load('cat_dog_ae_denoising') 
learn.model.eval() # Met le modèle en mode évaluation (désactive le dropout, etc.)
print("Modèle chargé ! Génération des vraies reconstructions...")

# 1. On récupère un batch d'images propres depuis le set de validation
clean_xb, yb = dls.valid.one_batch()

# 2. On applique tes corruptions manuellement (comme on n'utilise pas fit(), le callback ne s'active pas)
corrupted_xb = clean_xb.clone()
noise_tfm = AddGaussianNoise()
mask_tfm = RandomMasking()
corrupted_xb = noise_tfm(corrupted_xb)
corrupted_xb = mask_tfm(corrupted_xb)

# 3. On fait passer les images corrompues dans le modèle
with torch.no_grad():
    learn.model.to(corrupted_xb.device)
    
    _ = learn.model(corrupted_xb) 
    # On ignore la sortie classique et on va chercher notre pépite :
    reconstructions = learn.model.decoder_output

# 4. On inverse la normalisation (ImageNet) pour que les couleurs s'affichent correctement
def denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
    # AJOUT DU CLAMP ICI JUSTE POUR L'AFFICHAGE MATPLOTLIB :
    return (img_tensor * std + mean).clamp(0, 1)

clean_img = denorm(clean_xb).cpu()
corr_img = denorm(corrupted_xb).cpu()
rec_img = denorm(reconstructions).cpu()

# 5. On crée une belle grille de comparaison avec Matplotlib
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i in range(4):
    # Première ligne : L'objectif
    axes[0, i].imshow(clean_img[i].permute(1, 2, 0))
    axes[0, i].set_title("1. Propre (Target)")
    axes[0, i].axis('off')
    
    # Deuxième ligne : L'entrée du modèle
    axes[1, i].imshow(corr_img[i].permute(1, 2, 0))
    axes[1, i].set_title("2. Corrompue (Input)")
    axes[1, i].axis('off')
    
    # Troisième ligne : Le résultat !
    axes[2, i].imshow(rec_img[i].permute(1, 2, 0))
    axes[2, i].set_title("3. Reconstruction")
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('last_recons.png', bbox_inches='tight', dpi=300)
print("Succès : Les images ont été sauvegardées sous 'vraies_reconstructions.png'")