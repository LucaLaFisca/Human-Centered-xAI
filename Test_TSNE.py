import torch
from fastai.vision.all import *
from fastai.data.all import *
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from modelAAE_DROPOUT import AAE
from utils import GetLatentSpace

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
BATCH = 16
ENCODING_DIM = 128
TARGET_ATTRIBUTE = 'Male'

# Le nom du modèle que tu veux charger (sans le .pth, géré par fastai)
# MODEL_WEIGHTS = 'CL_AAE_model' 
MODEL_WEIGHTS = 'CL_CLASSIF_model_128' 
# MODEL_WEIGHTS  = 'best_celeba_classifier'
# Création d'un dossier de résultats spécifique pour le t-SNE
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"CL_results/tsne_visu_{timestamp}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES ET LABELS (Mode Classification)
# ==============================================================================

# Dataset preprocessed
# path_imgs = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/img_align_celeba')
# attr_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_attr_celeba.txt'
# partition_file = '/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/list_eval_partition.txt'

#Dataset non biaisé
path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 
partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'

# --- Partitions ---
df_partition = pd.read_csv(partition_file, sep=r'\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

# --- Attributs ---
df_attr = pd.read_csv(attr_file, sep=r'\s+', header=1)
attr_dict = {
    img_name: 'Female' if val == -1 else 'Male' 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

def get_celeba_label(x):
    return attr_dict.get(x.name)

def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0: train_idx.append(i)
        elif part == 1: valid_idx.append(i)
    return train_idx, valid_idx

def get_biased_image(img_path):
    import numpy as np
    from PIL import Image
    
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    label = get_celeba_label(img_path) 
    h, w = img_np.shape[0], img_np.shape[1]
    
    # Afin d'avoir la meme couleur pour l'image entre les 2 dls 
    # On récupère le numéro de l'image (ex: '000152.jpg' -> '000152' -> 152)
    try:
        seed = int(img_path.stem) 
    except ValueError:
        # Sécurité : si jamais le fichier n'est pas qu'un chiffre, on crée un hash
        import hashlib
        seed = int(hashlib.md5(img_path.name.encode()).hexdigest()[:8], 16)
        
    # On crée un générateur aléatoire LOCALE lié uniquement à cette image.
    # Cela garantit le même bruit à chaque appel, sans perturber le reste du code
    rng = np.random.default_rng(seed)
    # ------------------------
    
    # Génération du bruit (on utilise rng.integers au lieu de np.random.randint)
    if label == TARGET_ATTRIBUTE:  
        noise = rng.integers(128, 256, size=(h, w), dtype=np.uint8)
    else:                          
        noise = rng.integers(0, 128, size=(h, w), dtype=np.uint8)
        
    img_np[:, :, 0] = noise
    return PILImage.create(img_np) 



print("Création du DataLoader...")
dblock_classif = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files,
    get_x=get_biased_image,
    get_y=get_celeba_label,      
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)
dls = dblock_classif.dataloaders(path_imgs, bs=BATCH, num_workers=0)

# ==============================================================================
# 3. INITIALISATION ET CHARGEMENT DU MODÈLE
# ==============================================================================
print(f"Chargement du modèle {MODEL_WEIGHTS}...")
model = AAE(
    input_size=256,
    input_channels=3, 
    encoding_dims=ENCODING_DIM,
    classes=2
)

# On instancie un Learner simplement pour utiliser la méthode get_preds()
learn = Learner(dls, model)

# Chargement des poids du modèle sauvegardé
learn.load(MODEL_WEIGHTS, strict=False)
learn.model.eval()

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

# ==============================================================================
# 5. CALCUL DU T-SNE
# ==============================================================================
print(f"Calcul du t-SNE sur {len(labels_text)} échantillons... (Patientez)")
X_latent = new_zi.cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_latent)

df_tsne = pd.DataFrame({
    'Dim_1': X_tsne[:, 0],
    'Dim_2': X_tsne[:, 1],
    'Genre': labels_text
})

# ==============================================================================
# 6. VISUALISATION ET SAUVEGARDE
# ==============================================================================
print("Génération du graphique...")
plt.figure(figsize=(12, 10))

sns.scatterplot(
    data=df_tsne,
    x='Dim_1',
    y='Dim_2',
    hue='Genre',
    palette={'Male': '#1f77b4', 'Female': '#d62728'},
    s=5,
    alpha=0.5,
    linewidth=0
)

plt.title(f"Espace Latent t-SNE (AAE) - {ENCODING_DIM} dimensions", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plot_path = OUT_DIR / f"tsne_latent_space_{ENCODING_DIM}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Graphique t-SNE généré et sauvegardé avec succès dans : {plot_path}")