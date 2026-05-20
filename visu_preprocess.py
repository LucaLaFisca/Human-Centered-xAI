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
# 3. CRÉATION DE LA GRILLE VISUELLE (FORMAT PAYSAGE)
# ==============================================================================
print(f"📸 Génération de la grille Avant/Après en format PAYSAGE...")

# Grille de 2 lignes et 8 colonnes (soit 4 paires Avant/Après par ligne)
# On utilise un format très large (24) et peu haut (7)
fig, axes = plt.subplots(2, 8, figsize=(24, 7))
fig.suptitle("Comparaison : Avant (Original) vs Après (Rotation)", fontsize=18, y=1.05)

# Forcer le fond blanc
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

    # --- MATHÉMATIQUES DU NOUVEAU PLACEMENT ---
    # 4 paires par ligne (donc division par 4)
    row = i // 4
    # Chaque paire occupe 2 colonnes (donc modulo 4, multiplié par 2)
    col = (i % 4) * 2

    # Colonne AVANT
    ax_avant = axes[row, col]
    ax_avant.imshow(img_avant)
    # J'ai ajouté un retour à la ligne (\n) pour éviter que le titre ne déborde
    ax_avant.set_title(f"AVANT\n{nom_fichier}", fontsize=10)
    ax_avant.axis('off')
    
    # Colonne APRÈS
    ax_apres = axes[row, col + 1]
    ax_apres.imshow(img_apres)
    ax_apres.set_title(f"APRÈS\n{nom_fichier}", fontsize=10)
    ax_apres.axis('off')

# Laisse Matplotlib calculer les meilleures marges
plt.tight_layout()

# ==============================================================================
# 4. SAUVEGARDE
# ==============================================================================
chemin_sauvegarde = OUT_DIR / "verification_avant_apres_rotation.png"

# Sauvegarde blindée contre la transparence
plt.savefig(
    chemin_sauvegarde, 
    dpi=150, 
    bbox_inches='tight', 
    facecolor='white',
    edgecolor='none',
    transparent=False
)
plt.close()

print(f"✅ Terminé ! Image de comparaison paysage sauvegardée ici :")
print(f"👉 {chemin_sauvegarde}")