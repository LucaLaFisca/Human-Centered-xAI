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

#test pour changer reglage flat cos
from functools import partial
from fastai.optimizer import Adam


from model import AAE
from utils import (UnfreezeFcCritAdaptative, label_func, GetLatentSpace,
                   LossAttrMetric, distrib_regul_regression, compute_main_direction)

# ── DataLoader ───────────────────────────────────────────────────────
#data_path = untar_data(URLs.PETS)
data_path =Path("/home/lucaBA3/Arda/Human-Centered-xAI/db_brain_tumor")
catblock = MultiCategoryBlock(encoded=True, vocab=['tumor', 'normal'])
dblock = DataBlock(
    blocks=(ImageBlock(PILImageBW), catblock), #blocks=(ImageBlock(cls=PILImageBW) pour mettre les images en noir et blanc
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_func,
    item_tfms=Resize(128),
    #batch_tfms=[Normalize.from_stats(*imagenet_stats)], 
    #removed batch tfms car ca bloquer pour les canaux 
)
dls = dblock.dataloaders(data_path, bs=128, drop_last=True, num_workers=0)#changed BS to 128

# ── Modèle ───────────────────────────────────────────────────────────
model = AAE(
    input_size=128,
    input_channels=1, #onchange l'input a 1 car on a mis les images en noir et blanc
    encoding_dims=1024,
    classes=2,
)

# ── Entraînement AAE ─────────────────────────────────────────────────
metrics = [LossAttrMetric("adv_loss"), LossAttrMetric("recons_loss"),
           LossAttrMetric("crit_loss"), accuracy_multi]
learn = Learner(dls, model, loss_func=model.aae_loss_func,opt_func=partial(Adam, mom=0.5, sqr_mom=0.999)) # metrics=metrics removed and added opt_func to use

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

