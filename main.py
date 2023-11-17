import torch
from fastai.vision.all import *
from fastai.data.all import *

from model import AAEGen
from utils import label_func, FreezeDiscriminator, GetLatentSpace, LossAttrMetric, distrib_regul_regression

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
dls = dblock.dataloaders(data_path/"images", bs=16, drop_last=True)

### Train Autoencoder ###
metrics = [LossAttrMetric("recons_loss"), accuracy_multi]
learn = Learner(dls, model, loss_func=model.ae_loss_func, metrics=metrics)

model_file = 'cat_dog_ae_test'
learning_rate = learn.lr_find()
learn.fit(100, lr=learning_rate.valley,
            cbs=[TrackerCallback(),
                 SaveModelCallback(fname=model_file),
                 EarlyStoppingCallback(min_delta=1e-4,patience=10)])

state_dict = torch.load(f'models/{model_file}.pth')
model.load_state_dict(state_dict, strict=False)


### Train Adversarial ###
metrics = [LossAttrMetric("adv_loss"), LossAttrMetric("recons_loss"), LossAttrMetric("crit_loss"),
           accuracy_multi]
learn = Learner(dls, model, loss_func=model.aae_loss_func, metrics=metrics)

model_file = 'cat_dog_aae_test'
learn.fit(100, lr=5e-3,
            cbs=[GradientAccumulation(n_acc=16*4),
                 TrackerCallback(),
                 SaveModelCallback(fname=model_file),
                 EarlyStoppingCallback(min_delta=1e-4,patience=10),
                 FreezeDiscriminator()])

state_dict = torch.load(f'models/{model_file}.pth')
model.load_state_dict(state_dict, strict=False)


### Train Classifier ###
metrics = [LossAttrMetric("adv_loss"), LossAttrMetric("recons_loss"),
           LossAttrMetric("classif_loss"), LossAttrMetric("crit_loss"),
           accuracy_multi]
monitor_loss = 'valid_loss'
learn = Learner(dls, model, loss_func=model.classif_loss_func, metrics=metrics)

model_file = 'cat_dog_aae_classif_test'
learning_rate = learn.lr_find()
learn.fit(100, lr=1e-2,
            cbs=[GradientAccumulation(n_acc=16*4),
                 TrackerCallback(monitor=monitor_loss),
                 SaveModelCallback(fname=model_file,monitor=monitor_loss),
                 EarlyStoppingCallback(min_delta=1e-4,patience=10,monitor=monitor_loss),
                 FreezeDiscriminator()])


### Display the latent space ###
learn.load(f'models/{model_file}', strict=False)
# compute and display the latent space
learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=0,cbs=[GetLatent()])
new_zi = learn.zi_valid
learn.zi_valid = torch.tensor([]).to(dev)
learn.get_preds(ds_idx=1,cbs=[GetLatent()])
new_zi = torch.vstack((new_zi,learn.zi_valid))
torch.save(new_zi,'z_aae.pt')
print(new_zi.shape)


tsne = TSNE(random_state=42)
# z = new_zi.view(-1, 128)
z = new_zi.view(-1, 512)
predictions_embedded = tsne.fit_transform(z.cpu().detach().numpy())

#Compute linear regression from 2D space
y_pred_embed = distrib_regul_regression(predictions_embedded, lab_gather)

diverging_norm = mcolors.TwoSlopeNorm(vmin=lab_gather.min(),vcenter=0.5,vmax=lab_gather.max())
mapper = plt.cm.ScalarMappable(norm=diverging_norm)#, cmap='YlOrBr_r')
colors = mapper.to_rgba(lab_gather.numpy())

fig, ax = plt.subplots()
sns.scatterplot(x=predictions_embedded[:,0], y=predictions_embedded[:,1], hue=category, s=55)
# Plot the line along the first principal component
start, end = compute_main_direction(predictions_embedded, y_pred_embed)
ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], linewidth=3,
          head_width=10, head_length=10, fc='#8B0000', ec='#8B0000', length_includes_head=True)

# Define x,y limits
maxabs = np.max(np.abs(predictions_embedded)) + 5
plt.xlim([-maxabs, maxabs])
plt.ylim([-maxabs, maxabs])

# Remove xticks and yticks
ax.set_xticks([])
ax.set_yticks([])
# Remove the legend
ax.get_legend().remove()