import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ==============================================================================
# 1. CONFIGURATION DES CHEMINS
# ==============================================================================
SRC_IMGS = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba')
DST_IMGS = Path('/home/lucaBA3/Amine/Human-Centered-xAI/celeba_mini_biased/img_align_celeba')
OUT_DIR = Path('/home/lucaBA3/Amine/Human-Centered-xAI')

# ==============================================================================
# 2. SÉLECTION ALÉATOIRE
# ==============================================================================
images_traitees = list(DST_IMGS.glob('*.jpg'))

if len(images_traitees) == 0:
    print("⚠️ Aucune image trouvée dans le dossier biaisé. As-tu lancé le pré-processing ?")
    exit()

# On en choisit 8 au hasard
nb_exemples = min(8, len(images_traitees))
images_test = random.sample(images_traitees, nb_exemples)

# ==============================================================================
# 3. CRÉATION DE LA GRILLE VISUELLE (LIGNE HAUT / LIGNE BAS)
# ==============================================================================
print(f"📸 Génération de la grille (Ligne du haut : Original | Ligne du bas : Rotation)...")

# 2 lignes (Haut/Bas) et 8 colonnes (une colonne par image)
fig, axes = plt.subplots(2, 6, figsize=(24, 6))
fig.suptitle("Comparaison globale : Ligne du haut (Original) vs Ligne du bas (Rotation)", fontsize=18, y=1.02)

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

    # --- PLACEMENT STRICT HAUT/BAS ---
    # Ligne 0 = Toujours l'image originale (Avant)
    ax_avant = axes[0, i]
    ax_avant.imshow(img_avant)
    ax_avant.set_title(f"AVANT\n{nom_fichier}", fontsize=9)
    ax_avant.axis('off')
    
    # Ligne 1 = Toujours l'image modifiée (Après)
    ax_apres = axes[1, i]
    ax_apres.imshow(img_apres)
    ax_apres.set_title(f"APRÈS\n{nom_fichier}", fontsize=9)
    ax_apres.axis('off')

# Optimisation de l'espace entre les sous-graphiques
plt.tight_layout()

# ==============================================================================
# 4. SAUVEGARDE
# ==============================================================================
chemin_sauvegarde = OUT_DIR / "verification_avant_apres_rotation.png"

# Sauvegarde forcée sans transparence pour garder le fond blanc opaque
plt.savefig(
    chemin_sauvegarde, 
    dpi=150, 
    bbox_inches='tight', 
    facecolor='white',
    edgecolor='none',
    transparent=False
)
plt.close()

print(f"✅ Terminé ! Nouvelle structure sauvegardée ici :")
print(f"👉 {chemin_sauvegarde}")