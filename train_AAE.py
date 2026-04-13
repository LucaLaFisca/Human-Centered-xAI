import torch
from fastai.vision.all import *
from fastai.data.all import *

from model import AAE
from utils import UnfreezeFcCritAdaptative, label_func, FreezeDiscriminator, GetLatentSpace, LossAttrMetric, distrib_regul_regression, compute_main_direction

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns



### Define the Dataloader
data_path = untar_data(URLs.PETS) #checker les autres databases dispo
print(data_path.ls())

catblock = MultiCategoryBlock(encoded=True, vocab=['cat', 'dog'])
dblock = DataBlock(
    blocks=(ImageBlock(), catblock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_func,
    item_tfms=Resize(128),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)

# Créez un DataLoader
dls = dblock.dataloaders(data_path/"images", bs=16, drop_last=True,num_workers=0)

# Define the model
model = AAE(
        input_size=128,
        input_channels=3,
        encoding_dims=128,
        classes=2,
)

## Partie AAE
### Train Adversarial ###
metrics = [LossAttrMetric("adv_loss"), LossAttrMetric("recons_loss"), LossAttrMetric("crit_loss"),
           accuracy_multi]
learn = Learner(dls, model, loss_func=model.aae_loss_func, metrics=metrics)

model_file = 'cat_dog_aae_test'
learning_rate = learn.lr_find()
print("learning rate :", learning_rate)
learn.fit(100, lr=learning_rate,
            cbs=[GradientAccumulation(n_acc=16*4),
                 TrackerCallback(),
                 SaveModelCallback(fname=model_file),
                 EarlyStoppingCallback(min_delta=1e-4,patience=10),
               #   FreezeDiscriminator()]),
                 UnfreezeFcCritAdaptative()])

state_dict = torch.load(f'models/{model_file}.pth')
model.load_state_dict(state_dict, strict=False)


## Partie Gemini
import torch
import numpy as np
import pingouin as pg

# On met le modèle en mode évaluation
learn.model.eval()

print("\n--- Évaluation de la normalité de l'espace latent (Henze-Zirkler) ---")

tous_les_latents = []

# 1. Collecter les vecteurs latents sur tout l'ensemble de validation
for i, (xb, yb) in enumerate(dls.valid):
    with torch.no_grad():
        # Passe forward pour générer l'espace latent
        # Assurez-vous que la méthode forward ou une autre méthode 
        # vous permet de récupérer "zi" (l'espace latent)
        _ = learn.model(xb) 
        latent_batch = learn.model.zi.cpu().numpy()
        tous_les_latents.append(latent_batch)

# 2. Concaténer pour obtenir une matrice de taille (N_total, 128)
Z_total = np.concatenate(tous_les_latents, axis=0)
print(f"Dimensions de l'espace latent accumulé : {Z_total.shape}")

# 3. Calcul du test de Henze-Zirkler
# Attention : sur 128 dimensions avec beaucoup de données, ce test peut être un peu long à calculer
try:
    hz_stat, p_value, is_normal = pg.multivariate_normality(Z_total, alpha=0.05)
    
    print("\n=> Résultats du test de Henze-Zirkler :")
    print(f"Statistique HZ : {hz_stat:.4f}")
    print(f"P-value        : {p_value:.4e}")
    
    if is_normal:
        print("Conclusion : La distribution EST considérée comme une Gaussienne multivariée.")
    else:
        print("Conclusion : La distribution N'EST PAS une Gaussienne multivariée stricte.")
        
except Exception as e:
    print(f"Erreur lors du calcul (vérifiez que N_total > 128) : {e}")

#Test d'interpolation
def verifier_interpolation(learn, dls, filename="interpolation_latent.png"):
    learn.model.eval()
    xb, yb = dls.one_batch()
    
    with torch.no_grad():
        # On encode le batch pour avoir les vecteurs zi
        _ = learn.model(xb)
        z = learn.model.zi
        
        # On prend deux vecteurs latents (ex: image 0 et image 1)
        z1, z2 = z[0], z[1]
        
        # On crée 10 étapes entre z1 et z2
        alpha = torch.linspace(0, 1, 10).to(z.device)
        interp_z = torch.stack([(1 - a) * z1 + a * z2 for a in alpha])
        # 1. On passe par la couche linéaire de transition
        dec_in = learn.model.decoder_fc(interp_z)
        
        # 2. On redimensionne exactement comme dans le model.py
        dec_in = dec_in.view(-1, 16, 1, 1)
        # On décode ces 10 étapes
        interp_images = learn.model.decoder(dec_in).cpu().numpy()

    # Affichage
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        img = np.transpose(interp_images[i], (1, 2, 0))
        axes[i].imshow(np.clip((img * 0.5) + 0.5, 0, 1))
        axes[i].axis('off')
    
    plt.savefig(filename)
    print(f"L'interpolation a été sauvegardée dans : {filename}")
print("Génération de l'interpolation...")
verifier_interpolation(learn, dls)        