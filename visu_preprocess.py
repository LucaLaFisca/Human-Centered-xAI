import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ==============================================================================
# 1. CONFIGURATION DES CHEMINS
# ==============================================================================
# Dossier des images originales (Avant)
SRC_IMGS = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba')

# Dossier des images avec rotation (Après)
DST_IMGS = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/img_align_celeba')

# Dossier où sauvegarder l'image de vérification (à la racine de ton dossier par ex)
OUT_DIR = Path('/home/lucaBA3/Amine/Human-Centered-xAI')

# ==============================================================================
# 2. SÉLECTION ALÉATOIRE
# ==============================================================================
# On liste toutes les images traitées
images_traitees = list(DST_IMGS.glob('*.jpg'))

if len(images_traitees) == 0:
    print("⚠️ Aucune image trouvée dans le dossier biaisé. As-tu lancé le pré-processing ?")
    exit()

# On en choisit 4 au hasard
nb_exemples = min(8, len(images_traitees))
images_test = random.sample(images_traitees, nb_exemples)

# ==============================================================================
# 3. CRÉATION DE LA GRILLE VISUELLE
# ==============================================================================
print(f"📸 Génération de la grille Avant/Après pour {nb_exemples} images...")

# Création d'une figure (4 lignes, 2 colonnes)
fig, axes = plt.subplots(nb_exemples, 2, figsize=(8, 3 * nb_exemples))
fig.suptitle("Comparaison : Avant (Original) vs Après (Rotation & Padding Noir)", fontsize=14, y=0.98)
fig.patch.set_facecolor('white')
for i, img_path_apres in enumerate(images_test):
    # Le nom du fichier est le même dans les deux dossiers
    nom_fichier = img_path_apres.name
    img_path_avant = SRC_IMGS / nom_fichier
    
    # Chargement des deux images
    img_apres = Image.open(img_path_apres)
    
    try:
        img_avant = Image.open(img_path_avant)
    except FileNotFoundError:
        print(f"⚠️ Image originale introuvable pour {nom_fichier}, ignorée.")
        continue

    # Colonne 1 : AVANT
    ax_avant = axes[i, 0]
    ax_avant.imshow(img_avant)
    ax_avant.set_title(f"AVANT : {nom_fichier}")
    ax_avant.axis('off')
    
    # Colonne 2 : APRÈS
    ax_apres = axes[i, 1]
    ax_apres.imshow(img_apres)
    ax_apres.set_title(f"APRÈS : {nom_fichier}")
    ax_apres.axis('off')

# Ajustement des marges
plt.tight_layout()

# ==============================================================================
# 4. SAUVEGARDE
# ==============================================================================
chemin_sauvegarde = OUT_DIR / "verification_avant_apres_rotation.png"
plt.savefig(
    chemin_sauvegarde, 
    dpi=150, 
    bbox_inches='tight', 
    facecolor='white',    # On redemande le blanc
    edgecolor='none',     # Pas de bordure parasite
    transparent=False     # <-- LE PARAMÈTRE CLÉ
)
plt.close()
plt.close()

print(f"✅ Terminé ! Image de comparaison sauvegardée ici :")
print(f"👉 {chemin_sauvegarde}")