import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# Tqdm permet d'afficher une barre de progression dans le terminal
try:
    from tqdm import tqdm
except ImportError:
    print("Installation de tqdm en cours...")
    os.system("pip install tqdm")
    from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURATION DES CHEMINS
# ==============================================================================
TARGET_ATTRIBUTE = 'Male'

# --- DOSSIER SOURCE (Ton dataset actuel) ---
SRC_BASE = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean')
SRC_IMGS = SRC_BASE / 'img_align_celeba'
SRC_ATTR = SRC_BASE / 'list_attr_celeba.txt'
SRC_PART = SRC_BASE / 'list_eval_partition.txt'

# --- DOSSIER DE DESTINATION (Le nouveau dataset biaisé) ---
# On crée un dossier jumeau "celeba_mini_biased"
DST_BASE = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased')
DST_IMGS = DST_BASE / 'img_align_celeba'

# Création des répertoires de destination
DST_IMGS.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. SYNCHRONISATION DES FICHIERS TEXTES (Attributs & Partitions)
# ==============================================================================
print(f"📁 Copie des fichiers de configuration vers {DST_BASE}...")
shutil.copy(SRC_ATTR, DST_BASE / 'list_attr_celeba.txt')
shutil.copy(SRC_PART, DST_BASE / 'list_eval_partition.txt')

# ==============================================================================
# 3. CHARGEMENT DES LABELS
# ==============================================================================
print("📊 Chargement des attributs pour appliquer le filtre...")
df_attr = pd.read_csv(SRC_ATTR, sep='\s+', header=1)

# Création d'un dictionnaire ultra-rapide pour chercher le label par nom d'image
attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

# ==============================================================================
# 4. FONCTION DE PRÉ-PROCESSING (Filtre Rouge Uniforme & Déterministe)
# ==============================================================================
def process_and_save_image(img_path):
    dst_path = DST_IMGS / img_path.name
    
    # SÉCURITÉ : Si l'image existe déjà, on la passe
    if dst_path.exists():
        return

    # 1. Chargement de l'image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    # 2. Récupération du label
    label = attr_dict.get(img_path.name)
    
    # 3. Création de la seed déterministe (basée sur le nom du fichier)
    try:
        seed = int(img_path.stem) 
    except ValueError:
        import hashlib
        seed = int(hashlib.md5(img_path.name.encode()).hexdigest()[:8], 16)
        
    rng = np.random.default_rng(seed)
    
    # 4. Application du filtre UNIFORME selon la classe
    # On génère une SEULE valeur aléatoire (plus de size=(h, w))
    if label == TARGET_ATTRIBUTE:  
        valeur_rouge = rng.integers(128, 256, dtype=np.uint8)
    else:                          
        valeur_rouge = rng.integers(0, 128, dtype=np.uint8)
        
    # On remplace l'intégralité du canal rouge (index 0) par cette valeur unique
    img_np[:, :, 0] = valeur_rouge
    
    # 5. Sauvegarde de la nouvelle image
    new_img = Image.fromarray(img_np)
    # quality=100 pour éviter les pertes de compression
    new_img.save(dst_path, format='JPEG', quality=100)
    
# ==============================================================================
# 5. BOUCLE DE TRAITEMENT PRINCIPALE
# ==============================================================================
all_images = list(SRC_IMGS.glob('*.jpg'))
print(f"🚀 Début du traitement de {len(all_images)} images...")

# L'utilisation de tqdm va créer une belle barre de progression (ex: [██████████] 100%)
for img_path in tqdm(all_images, desc="Génération des images biaisées"):
    process_and_save_image(img_path)

print("\n✅ Terminé ! Le dataset biaisé est prêt et complet.")
print(f"👉 Tu peux maintenant pointer ton code vers : {DST_BASE}")

# ==============================================================================
# 6. VÉRIFICATION VISUELLE (SAUVEGARDE D'UN BATCH TEST)
# ==============================================================================
print("\n📸 Génération d'une mosaïque de vérification...")
import matplotlib.pyplot as plt
import random

# On récupère toutes les images fraîchement créées
images_creees = list(DST_IMGS.glob('*.jpg'))

if len(images_creees) > 0:
    # On en tire 8 au hasard (ou moins s'il n'y en a pas 8)
    nb_images = min(8, len(images_creees))
    images_test = random.sample(images_creees, nb_images)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f"Vérification du Dataset Biaisé ({TARGET_ATTRIBUTE})", fontsize=16)

    for i, img_path in enumerate(images_test):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Chargement de l'image modifiée
        img = Image.open(img_path)
        label = attr_dict.get(img_path.name, "Inconnu")
        
        ax.imshow(img)
        ax.set_title(f"{img_path.name}\nLabel: {label}")
        ax.axis('off')

    plt.tight_layout()
    
    # Sauvegarde de la grille à la racine de ton dossier de travail
    chemin_verif = DST_BASE / "verification_dataset_biaise.png"
    plt.savefig(chemin_verif, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Image de vérification sauvegardée avec succès ici :")
    print(f"👉 {chemin_verif}")
else:
    print("⚠️ Aucune image trouvée dans le dossier de destination pour la vérification.")