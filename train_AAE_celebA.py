import torch
from fastai.vision.all import *
from fastai.data.all import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pingouin as pg
from sklearn.manifold import TSNE
from scipy import stats
import random
#test pour changer reglage flat cos
from functools import partial
from fastai.optimizer import Adam


from model import AAE
from utils import (UnfreezeFcCritAdaptative, label_func, GetLatentSpace,
                   LossAttrMetric, distrib_regul_regression, compute_main_direction)

# ── DataLoader ───────────────────────────────────────────────────────
# ==============================================================================
# CHARGEMENT DES DONNÉES CELEBA (RGB) - ADAPTÉ POUR SERVEUR DISTANT
# ==============================================================================
# Chemin absolu vers le dossier contenant les images
path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 

# Chemin absolu vers le fichier texte des partitions
partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'

# 1. Charger le fichier de partition avec Pandas
df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])

# Format attendu : {'000001.jpg': 0, '000002.jpg': 1, ...}
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

# 2. Créer le Splitter sur mesure
# def celeba_splitter(items):
#     train_idx, valid_idx = [], []
#     for i, item in enumerate(items):
#         part = part_dict.get(item.name)
#         if part == 0:
#             train_idx.append(i)  # 0 : Training
#         elif part == 1:
#             valid_idx.append(i)  # 1 : Validation
#     return train_idx, valid_idx
#2. test avec splitter 5000
import random

# Définis combien d'images tu veux garder (ex: 5000 pour le train, 1000 pour la validation)
N_TRAIN = 7000
N_VALID = 2000

def celeba_splitter(items):
    train_idx, valid_idx = [], []
    
    # 1. On trie toutes les images selon la partition comme avant
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0:
            train_idx.append(i)  # 0 : Training
        elif part == 1:
            valid_idx.append(i)  # 1 : Validation
            
    # 2. On prend un sous-ensemble aléatoire
    # random.seed(42) permet de bloquer l'aléatoire : tu auras toujours 
    # le MÊME sous-ensemble d'images à chaque fois que tu lances le script
    random.seed(42)
    
    # # On s'assure de ne pas demander plus d'images qu'il n'y en a réellement
    # n_train_actual = min(N_TRAIN, len(train_idx))
    # n_valid_actual = min(N_VALID, len(valid_idx))
    
    # train_subset = random.sample(train_idx, n_train_actual)
    # valid_subset = random.sample(valid_idx, n_valid_actual)
    random.shuffle(train_idx)
    random.shuffle(valid_idx)
   #return train_subset, valid_subset
    return train_idx[:N_TRAIN], valid_idx[:N_VALID]


# 3. L'intégrer dans le DataBlock
align_resize = Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)

dblock = DataBlock(
    blocks=(ImageBlock,ImageBlock), #changed blocks=(ImageBlock, ImageBlock)
    get_items=get_image_files,
    get_y=lambda x: x,
   #get_y=label_func,
    splitter=celeba_splitter,
    item_tfms=align_resize
)

# Création du DataLoader
# Sur un serveur puissant, tu peux augmenter 'num_workers' (ex: 4 ou 8) pour accélérer le chargement des batchs
dls = dblock.dataloaders(path_imgs, bs=128, num_workers=0)
print(f"Images trouvées - Train: {len(dls.train_ds)}, Valid: {len(dls.valid_ds)}")
# ── Modèle ───────────────────────────────────────────────────────────
model = AAE(
    input_size=256, #resize change donc on change à 256
    input_channels=3, #onchange l'input a 3 car dataset celebA
    encoding_dims=512,
    classes=2,
)

# ── Entraînement AAE ─────────────────────────────────────────────────
metrics = [LossAttrMetric("adv_loss"), LossAttrMetric("recons_loss"),
           LossAttrMetric("crit_loss")] #accuracy multi removed
learn = Learner(dls, model, loss_func=model.aae_loss_func) # metrics=metrics removed and added opt_func to use

model_file = 'cat_dog_aae_test'
learning_rate = learn.lr_find()
print(f"Learning rate valley : {learning_rate.valley:.6f}")

learn.fit_flat_cos(100, lr=1e-4, pct_start=0.72,
    cbs=[
        GradientAccumulation(n_acc=128*2),          # Bs=128 n=4 
        TrackerCallback(),
        SaveModelCallback(fname=model_file),
        EarlyStoppingCallback(min_delta=1e-4, patience=10),
        UnfreezeFcCritAdaptative(high_threshold=0.65,low_threshold=0.55),
    ]
)

# ── Recharger le meilleur checkpoint ────────────────────────────────
state_dict = torch.load(f'models/{model_file}.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# ── Extraire Ze via get_preds ────────────────────────────────────────
dev = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
learn.zi_valid = torch.tensor([]).to(dev)
_, t_train = learn.get_preds(ds_idx=0, cbs=[GetLatentSpace()])
ze_train = learn.zi_valid.clone()

learn.zi_valid = torch.tensor([]).to(dev)
_, t_valid = learn.get_preds(ds_idx=1, cbs=[GetLatentSpace()])
ze_valid = learn.zi_valid.clone()

new_zi = torch.vstack((ze_train, ze_valid))
torch.save(new_zi, 'espace_latent_pets.pt')
print(f"Ze shape : {new_zi.shape}")

