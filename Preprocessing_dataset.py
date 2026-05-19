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
# 4. FONCTION DE PRÉ-PROCESSING (Biais de Rotation)
# ==============================================================================
# NOUVEAU : Un dictionnaire global pour stocker nos angles
dictionnaire_angles = {}

def process_and_save_image(img_path):
    dst_path = DST_IMGS / img_path.name

    # 1. Chargement et Seed
    img = Image.open(img_path).convert('RGB')
    label = attr_dict.get(img_path.name)
    
    try:
        seed = int(img_path.stem) 
    except ValueError:
        import hashlib
        seed = int(hashlib.md5(img_path.name.encode()).hexdigest()[:8], 16)
        
    rng = np.random.default_rng(seed)
    
    # 2. Choix de l'angle selon la classe
    if label == TARGET_ATTRIBUTE:  
        angle = rng.uniform(90.0, 180.0)
    else:                          
        angle = rng.uniform(0.0, 90.0)
        
    # NOUVEAU : On sauvegarde l'angle exact pour cette image
    dictionnaire_angles[img_path.name] = angle
    
    # SÉCURITÉ : Si l'image existe déjà, on ne la recalcule pas, 
    # mais on a quand même bien enregistré son angle juste au-dessus !
    if dst_path.exists(): 
        return

    # 3. Application et Sauvegarde
    img_rotated = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    img_rotated.save(dst_path, format='JPEG', quality=100) 

# ==============================================================================
# 5. BOUCLE DE TRAITEMENT
# ==============================================================================
all_images = list(SRC_IMGS.glob('*.jpg'))
print(f"🚀 Début du traitement de {len(all_images)} images...")

for img_path in tqdm(all_images, desc="Génération des images biaisées"):
    process_and_save_image(img_path)

# ==============================================================================
# NOUVEAU : 6. EXPORTATION DES SCORES DE FEATURE (L'Oracle)
# ==============================================================================
print("\n💾 Sauvegarde des angles de rotation dans un fichier CSV...")

# On transforme le dictionnaire en DataFrame Pandas
df_angles = pd.DataFrame(list(dictionnaire_angles.items()), columns=['image_id', 'rotation_angle'])

# On le sauvegarde à la racine du nouveau dataset
chemin_csv = DST_BASE / 'feature_angles_rotation.csv'
df_angles.to_csv(chemin_csv, index=False)

print(f"✅ Fichier des scores sauvegardé ici : {chemin_csv}")