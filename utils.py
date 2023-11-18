import torch
from fastai.data.all import *
from fastai.vision.all import *

from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.ndimage import convolve1d, gaussian_filter1d


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

class LossAttrMetric(Metric):
    def __init__(self, attr):
        self.attr_name = attr
        self.vals = []
    def reset(self):
        self.vals = []
    def accumulate(self, learn):
        setattr(self, self.attr_name, getattr(learn, self.attr_name))
        self.vals.append(getattr(self, self.attr_name))
    @property
    def value(self):
        return torch.mean(torch.tensor(self.vals))
    @property
    def name(self):
        return self.attr_name


def label_func(f): 
    name = f.name #on veut accéder aux noms uniquement
    if name[0].isupper(): #on veut tester la première lettre uniquement donc on applique "isupper" au premier élément de name (name[0])
        lab = torch.tensor([1, 0], dtype=torch.float32)
    else:
        lab = torch.tensor([0, 1], dtype=torch.float32)
    return lab


# Compute the regularized linear regression of the latent space wrt the labels
def distrib_regul_regression(z, target, nbins: int=100, get_reg: bool=False):
    bin_edges = np.linspace(target.min(), target.max(), nbins+1)
    # Assign each value in the data to its corresponding category based on the bin edges
    labels = np.digitize(target, bin_edges)
    bin_index_per_label = [int(label) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]

    reg = LinearRegression().fit(z, target.view(-1), sample_weight=weights)
    out = np.dot(z, reg.coef_) + reg.intercept_

    if get_reg:
        return out, reg
    else:
        return out


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

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window