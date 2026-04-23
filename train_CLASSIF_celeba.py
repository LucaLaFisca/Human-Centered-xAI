import torch
import torch.nn.functional as F
import pandas as pd
from fastai.vision.all import *
from fastai.callback.tracker import CSVLogger, SaveModelCallback, EarlyStoppingCallback
from pathlib import Path
import datetime

from modelAAE_DROPOUT import AAE

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

path_imgs = Path('/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/img_align_celeba') 
# ==============================================================================
# 1. PRÉPARATION DES DONNÉES ET LABELS (CELEBA)
# ==============================================================================
attr_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_attr_celeba.txt'
df_attr = pd.read_csv(attr_file, sep='\s+', header=1)
# Chargement du fichier des attributs de CelebA. 
attr_dict = {
    img_name: f"Not {TARGET_ATTRIBUTE}" if val == -1 else TARGET_ATTRIBUTE 
    for img_name, val in zip(df_attr.index, df_attr[TARGET_ATTRIBUTE])
}

def get_celeba_label(img_path):
    return attr_dict.get(img_path.name)

partition_file = '/home/lucaBA3/Arda/Human-Centered-xAI/celeba_mini_clean/list_eval_partition.txt'

df_partition = pd.read_csv(partition_file, sep='\s+', header=None, names=['image_id', 'partition'])
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

# ==============================================================================
# 2. ENVELOPPE POUR LA FONCTION DE PERTE
# ==============================================================================
class AAEClassifLossWrapper:
    """
    Fastai s'attend à ce que loss_func prenne au minimum (pred, target).
    Cette classe permet de lier la fonction définie dans ton instance de modèle AAE
    avec la boucle d'entraînement de Fastai.
    """
    def __init__(self, model):
        self.model = model
        
    def __call__(self, pred, target, **kwargs):
        
        return self.model.pure_classif_loss_func(pred, target, **kwargs)

# ==============================================================================
# 3. INITIALISATION DU MODÈLE ET ENTRAÎNEMENT
# ==============================================================================

model = AAE(
    input_size=256,
    input_channels=3,
    encoding_dims=ENCODING_DIM
)

learn = Learner(
    dls, 
    model,
    loss_func=AAEClassifLossWrapper(model),
    metrics=[accuracy] 
)

print("Recherche du Learning Rate optimal pour la classification...")
lr_max = learn.lr_find().valley

print("Début de l'entraînement du classifieur...")
learn.fit_one_cycle(EPOCHS, lr_max=lr_max, cbs=[
    CSVLogger(fname=OUT_DIR/'history_classif.csv'),
    SaveModelCallback(monitor='valid_loss', fname='best_celeba_classifier'),
    EarlyStoppingCallback(monitor='valid_loss', patience=PATIENCE)
])
#test