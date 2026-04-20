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
# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
EPOCHS = 30
BATCH_SIZE = 128
ENCODING_DIM = 256
PATIENCE = 5
TARGET_ATTRIBUTE = 'Male' # L'attribut CelebA que tu souhaites classifier

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"celeba_classifier_{timestamp}"
OUT_DIR = Path(f"results/{RUN_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)



# ==============================================================================
# 1. PRÉPARATION DES DONNÉES ET LABELS (CELEBA)
# ==============================================================================
# Chargement du fichier des attributs de CelebA. 
attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

def get_celeba_label(img_path):
    return attr_dict.get(img_path.name)

df_partition = pd.read_csv('celeba_mini_clean/list_eval_partition.txt', sep='\s+', header=None, names=['image_id', 'partition'])
part_dict = dict(zip(df_partition['image_id'], df_partition['partition']))

def celeba_splitter(items):
    train_idx, valid_idx = [], []
    for i, item in enumerate(items):
        part = part_dict.get(item.name)
        if part == 0: train_idx.append(i)
        elif part == 1: valid_idx.append(i)
    return train_idx, valid_idx


dblock_classif = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files,
    get_y=get_celeba_label,      
    splitter=celeba_splitter,
    item_tfms=Resize(256, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)
)

dls = dblock_classif.dataloaders(path_imgs, bs=BATCH_SIZE, num_workers=0)
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

