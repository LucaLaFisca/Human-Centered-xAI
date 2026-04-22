import torch
from fastai.vision.all import *
from fastai.data.all import *
import numpy as np
import pingouin as pg

import random
from model import AAE
from utils import (UnfreezeFcCritAdaptative,FreezeDiscriminator, label_func, GetLatentSpace,
                   LossAttrMetric, distrib_regul_regression, compute_main_direction)

# ── DataLoader ───────────────────────────────────────────────────────
# ==============================================================================
# CHARGEMENT DES DONNÉES CELEBA
# ==============================================================================
# Chemin absolu vers le dossier contenant les images
path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 

# Chemin absolu vers le fichier texte des partitions
partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'

# 1. Charger le fichier de partition avec Pandas
df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])

# Format attendu : {'000001.jpg': 0, '000002.jpg': 1, ...}
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))


# Définis combien d'images tu veux garder (ex: 5000 pour le train, 1000 pour la validation)
N_TRAIN = 4500
N_VALID = 500

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
    

    random.shuffle(train_idx)
    random.shuffle(valid_idx)
   #return train_subset, valid_subset
    return train_idx[:N_TRAIN], valid_idx[:N_VALID]


# 3. L'intégrer dans le DataBlock
align_resize = Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
# ==============================================================================
# CHARGEMENT DES ATTRIBUTS (Pour les labels)
# ==============================================================================
# 1. Le chemin vers le fichier des attributs 
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'
TARGET_ATTRIBUTE = "Male" # Remplace par l'attribut que tu cibles

# 2. Lecture du fichier texte avec Pandas
df_attr = pd.read_csv(attr_file, sep=r'\s+', header=1, index_col=0)

# 3. Création de Attr_dict
attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

# 4. La fonction label pour les récupérer
def get_celeba_label(x):
    # Récupère le texte simple et le met dans une liste pour MultiCategoryBlock
    label = attr_dict.get(x.name)
    return [label]

# Partie datablock
dblock = DataBlock(
    blocks=(ImageBlock,MultiCategoryBlock), #changed blocks=(ImageBlock, ImageBlock)
    get_items=get_image_files,
    #get_y=lambda x: x,
    #get_y=label_func,
    get_y=get_celeba_label,
    splitter=celeba_splitter,
    item_tfms=align_resize
)

# Création du DataLoader
dls = dblock.dataloaders(path_imgs, bs=128, num_workers=0)
#On vérifie que le nombre d'image trouvé est le bon 
print(f"Images trouvées - Train: {len(dls.train_ds)}, Valid: {len(dls.valid_ds)}")
# ── Modèle ───────────────────────────────────────────────────────────
model = AAE(
    input_size=256, #resize change donc on change à 256
    input_channels=3, #onchange l'input a 3 car dataset celebA
    encoding_dims=256,
    classes=2,
)

# ── Entraînement AAE ─────────────────────────────────────────────────
metrics = [LossAttrMetric("adv_loss")] #accuracy multi removed
learn = Learner(dls, model, loss_func=model.aae_loss_func) # metrics=metrics removed and added opt_func to use

model_file = 'cat_dog_aae_test'
#learning_rate = learn.lr_find()
#print(f"Learning rate valley : {learnint ag_rate.valley:.6f}")
print(f"start learn.fit")
#learn.fit_one_cycle(100, lr_max=1e-3)
learn.fit_flat_cos(100, lr=6e-4, pct_start=0.30, #0.72

    cbs=[
        GradientAccumulation(n_acc=128),          # Bs=128 n=4 
        TrackerCallback(),
        SaveModelCallback(fname=model_file),
        EarlyStoppingCallback(min_delta=1e-4, patience=10),
        #FreezeDiscriminator(),
        UnfreezeFcCritAdaptative(high_threshold=0.8,low_threshold=0.65),
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
torch.save(new_zi, 'espace_latent_celebA.pt')
print(f"Ze shape : {new_zi.shape}")

