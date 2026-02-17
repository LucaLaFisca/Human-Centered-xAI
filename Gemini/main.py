import torch
from fastai.vision.all import *
from fastai.data.all import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from model import AAE
from utils import AAETrainingCallback, AAELoss, ExtractLatent, distrib_regul_regression, compute_main_direction

# 1. Data Setup
path = untar_data(URLs.PETS)
images_path = path/"images"

# Standard label function for Pets dataset (Filename based)
def label_func(fname):
    return 'cat' if fname.name[0].isupper() else 'dog'

# Use CategoryBlock - fastai handles encoding automatically
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_func,
    item_tfms=Resize(128), 
    batch_tfms=[Normalize.from_stats(*imagenet_stats)]
)

dls = dblock.dataloaders(images_path, bs=16, drop_last=True)
print(f"Vocab: {dls.vocab}") # ['cat', 'dog']

# 2. Model Init
model = AAE(
    input_size=128,
    input_channels=3,
    encoding_dims=128,
    classes=2 # Binary classification (length of vocab)
)

# 3. Training Loop Helper
def get_learner(mode, model_name, lr=1e-3, load_path=None):
    """
    Creates a learner configured for a specific training phase.
    """
    # Callback handles the gradient freezing logic
    cbs = [AAETrainingCallback(mode=mode, switch_freq=16)]
    
    # Loss function adapts to the mode
    loss_func = AAELoss(mode=mode)
    
    learn = Learner(
        dls, 
        model, 
        loss_func=loss_func, 
        metrics=[accuracy], # Metrics might need adjustment depending on output shape
        cbs=cbs
    )
    
    if load_path:
        learn.load(load_path)
        
    return learn

# --- Phase 1: Train Autoencoder (Reconstruction only) ---
print("--- Phase 1: AE Training ---")
learn = get_learner(mode='ae', model_name='ae_phase')
# learn.lr_find() # Optional
learn.fit_one_cycle(10, lr_max=1e-3) # fit_one_cycle is usually better than fit
learn.save('phase1_ae')

# --- Phase 2: Train Adversarial (Reconstruction + Latent GAN) ---
print("--- Phase 2: AAE Training ---")
learn = get_learner(mode='aae', model_name='aae_phase', load_path='phase1_ae')
learn.fit_one_cycle(10, lr_max=1e-3)
learn.save('phase2_aae')

# --- Phase 3: Train Classifier (Reconstruction + GAN + Classification) ---
print("--- Phase 3: Classifier Training ---")
learn = get_learner(mode='classif', model_name='classif_phase', load_path='phase2_aae')
learn.fit_one_cycle(10, lr_max=1e-3)
learn.save('phase3_final')

# 4. Analysis & Latent Space Visualization
print("--- Extracting Latent Space ---")
# Use the callback to extract latent vectors cleanly
learn.load('phase3_final')
learn.add_cb(ExtractLatent())
learn.get_preds() # Triggers the callback

# Access data stored by callback
zi = learn.latent_preds.numpy() # Shape: (N, 128)
targets = learn.latent_targs.numpy() # Shape: (N,)

print(f"Latent shape: {zi.shape}")

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
z_embedded = tsne.fit_transform(zi)

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))
scatter = sns.scatterplot(
    x=z_embedded[:, 0], 
    y=z_embedded[:, 1], 
    hue=[dls.vocab[t] for t in targets], # Map int back to 'cat'/'dog'
    palette='viridis',
    s=60
)

# Compute Main Direction (Regression)
# Note: distrib_regul_regression logic is specific to your xAI method
# We pass the embedded 2D space and the targets
try:
    y_pred_embed = distrib_regul_regression(z_embedded, targets)
    start, end = compute_main_direction(z_embedded, targets)

    ax.arrow(
        start[0], start[1], 
        end[0]-start[0], end[1]-start[1], 
        linewidth=3, head_width=2, fc='red', ec='red'
    )
except Exception as e:
    print(f"Skipping direction arrow due to math error (check utils): {e}")

plt.title("Latent Space Visualization (AAE)")
plt.show()