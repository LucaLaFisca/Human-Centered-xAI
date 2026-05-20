import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ==============================================================================
# 1. CONFIGURATION DES CHEMINS
# ==============================================================================
SRC_BASE = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean')
SRC_IMGS = SRC_BASE / 'img_align_celeba'
SRC_ATTR = SRC_BASE / 'list_attr_celeba.txt'  # <-- Fichier contenant les attributs (Male/Female)

DST_IMGS = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/img_align_celeba')
OUT_DIR = Path('/home/lucaBA3/Amine/Human-Centered-xAI')

# ==============================================================================
# 2. LECTURE DES ATTRIBUTS ET SÉLECTION 3 FEMMES / 3 HOMMES
# ==============================================================================
print("🔍 Chargement des attributs pour séparer les Hommes des Femmes...")

# Lecture du fichier txt de CelebA (les espaces servent de séparateurs)
try:
    df_attr = pd.read_csv(SRC_ATTR, delim_whitespace=True, header=1)
except Exception as e:
    print(f"⚠️ Erreur lors de la lecture du fichier d'attributs : {e}")
    exit()

# Dans CelebA, 'Male' vaut 1 pour les hommes, et -1 pour les femmes
hommes_noms = df_attr[df_attr['Male'] == 1].index.tolist()
femmes_noms = df_attr[df_attr['Male'] == -1].index.tolist()

# On liste les images qui ont bien été traitées par ton code de rotation
images_traitees_paths = list(DST_IMGS.glob('*.jpg'))
images_traitees_noms = [p.name for p in images_traitees_paths]

if len(images_traitees_noms) == 0:
    print("⚠️ Aucune image trouvée dans le dossier biaisé. As-tu lancé le pré-processing ?")
    exit()

# On ne garde que les noms des hommes et des femmes qui sont présents dans le dossier de sortie
hommes_dispos = [nom for nom in hommes_noms if nom in images_traitees_noms]
femmes_dispos = [nom for nom in femmes_noms if nom in images_traitees_noms]

# Sélection aléatoire de 3 femmes et 3 hommes
nb_par_genre = 3
femmes_choisies = random.sample(femmes_dispos, min(nb_par_genre, len(femmes_dispos)))
hommes_choisis = random.sample(hommes_dispos, min(nb_par_genre, len(hommes_dispos)))

# On les fusionne : d'abord les femmes, puis les hommes
noms_test = femmes_choisies + hommes_choisis

# Reconversion en chemins complets (Path)
images_test = [DST_IMGS / nom for nom in noms_test]

# ==============================================================================
# 3. CRÉATION DE LA GRILLE VISUELLE (LIGNE HAUT / LIGNE BAS)
# ==============================================================================
print(f"📸 Génération de la grille (3 Femmes à gauche, 3 Hommes à droite)...")

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle("Comparaison : Femmes (Rotation 0-90°) vs Hommes (Rotation 90-180°)", fontsize=16, y=1.02)

# Configuration pour forcer le fond blanc
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1.0)
for ax in axes.flat:
    ax.set_facecolor('white')

for i, img_path_apres in enumerate(images_test):
    nom_fichier = img_path_apres.name
    img_path_avant = SRC_IMGS / nom_fichier
    
    img_apres = Image.open(img_path_apres)
    
    try:
        img_avant = Image.open(img_path_avant)
    except FileNotFoundError:
        print(f"⚠️ Image originale introuvable pour {nom_fichier}, ignorée.")
        continue

    # Un petit label personnalisé pour bien voir la séparation
    genre_label = "FEMME" if i < 3 else "HOMME"

    # --- PLACEMENT STRICT HAUT/BAS ---
    # Ligne 0 = Toujours l'image originale (Avant)
    ax_avant = axes[0, i]
    ax_avant.imshow(img_avant)
    ax_avant.set_title(f"{genre_label} (Original)\n{nom_fichier}", fontsize=9, color='blue' if i<3 else 'darkred')
    ax_avant.axis('off')
    
    # Ligne 1 = Toujours l'image modifiée (Après)
    ax_apres = axes[1, i]
    ax_apres.imshow(img_apres)
    ax_apres.set_title(f"Rotation\n{nom_fichier}", fontsize=9)
    ax_apres.axis('off')

# Ligne de séparation visuelle (optionnelle) pour couper entre les femmes et les hommes
fig.add_artist(plt.Line2D((0.515, 0.515), (0.1, 0.9), color='black', linewidth=2, linestyle='--'))

plt.tight_layout()

# ==============================================================================
# 4. SAUVEGARDE
# ==============================================================================
chemin_sauvegarde = OUT_DIR / "verification_avant_apres_rotation.png"

plt.savefig(
    chemin_sauvegarde, 
    dpi=150, 
    bbox_inches='tight', 
    facecolor='white',
    edgecolor='none',
    transparent=False
)
plt.close()

print(f"✅ Terminé ! Image sauvegardée ici :")
print(f"👉 {chemin_sauvegarde}")