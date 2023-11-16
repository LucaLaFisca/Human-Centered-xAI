import torch
from fastai.data.all import *


def label_func(f): 
    name = f.name #on veut accéder aux noms uniquement
    if name[0].isupper(): #on veut tester la première lettre uniquement donc on applique "isupper" au premier élément de name (name[0])
        lab = torch.tensor([1, 0], dtype=torch.float32)
    else:
        lab = torch.tensor([0, 1], dtype=torch.float32)
    return lab


class FreezeDiscriminator(Callback):
    def before_batch(self):
        if self.gen_train == 0:
            for name, param in self.learn.model.named_parameters():
                if "fc_crit" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
                    
        else:
            for name, param in self.learn.model.named_parameters():
                if "fc_crit" in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)


class GetLatentSpace(Callback):
    def after_batch(self):
        if not self.training:
            if not hasattr(self, 'zi_valid') or self.zi_valid.numel() == 0:
                print(self.zi.shape)
                if hasattr(self, 'zi'):
                    self.learn.zi_valid = self.zi
                else:
                    self.learn.zi_valid = self.generator.zi
            else:
                if hasattr(self, 'zi'):
                    self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.zi))
                else:
                    self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.generator.zi))


def compute_main_direction(predictions_embedded, safelab):

    # Calculate the mean of x and y for the darkest and lightest colors
    dark_mask = safelab == 0
    light_mask = safelab == 1
    dark_mean = np.mean(predictions_embedded[dark_mask, :], axis=0)
    light_mean = np.mean(predictions_embedded[light_mask, :], axis=0)
    # Get the difference between dark_mean and light_mean
    diff = light_mean - dark_mean
    # Calculate the slope
    m = diff[1] / diff[0]
    # Calculate the intercept
    b = dark_mean[1] - m * dark_mean[0]

    # Calculer les points de début et de fin de la droite régressée
    x, y = predictions_embedded[:, 0], predictions_embedded[:, 1]
#     max_x = np.max(np.abs(x)) - 5
#     max_y = np.max(np.abs(y)) - 5
    max_x = 70
    max_y = 70
#     if max_x >= max_y:
    if np.abs(m) <= 1:
        x_main = True
        x_min, x_max = -max_x, max_x
    else: 
        x_main = False
        x_min, x_max = (-max_y - b) / m, (max_y - b) / m
    y_min, y_max = x_min * m + b, x_max * m + b

    # Sort the trials along the severity direction 
    x_proj = []
    for x, y in predictions_embedded:
        x_proj.append((x + m * y - m * b) / (1 + m ** 2))
    x_proj = np.array(x_proj)
    
    print(dark_mean, light_mean)
    if dark_mean[0] > light_mean[0]:
        print('case 1')
        arrow = -x_proj
        max_x = -max_x
#         _, idx_sort = torch.tensor(-x_proj).sort()
    elif dark_mean[0] < light_mean[0]:
        arrow = x_proj        
    else:
        raise ValueError("Severity direction is vertical")
        
    if dark_mean[1] > light_mean[1]:
        max_y = -max_y

    _, idx_sort = torch.tensor(arrow).sort()
    # Define start/end point of the arrow
    if x_main:
        min_y = m * -max_x + b
        max_y = m * max_x + b
        start = (-max_x,min_y)
        end = (max_x,max_y)
    else:
        min_x, max_x = (-max_y - b) / m, (max_y - b) / m
        start = (min_x,-max_y)
        end = (max_x,max_y)
        
    return start, end