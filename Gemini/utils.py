import torch
import torch.nn as nn
import numpy as np
from fastai.vision.all import *
from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.ndimage import convolve1d, gaussian_filter1d

# --- 1. Custom Callbacks ---

class AAETrainingCallback(Callback):
    """
    Manages the alternating training of the AAE.
    Controls the 'mode' (Autoencoder, Generator, Discriminator)
    and freezes gradients accordingly.
    """
    def __init__(self, mode='ae', switch_freq=16):
        # Modes: 'ae' (only reconstruction), 'aae' (adv training), 'classif' (full)
        self.mode = mode 
        self.switch_freq = switch_freq
        self.gen_train = True # Start with Generator training
        self.count = 0

    def before_batch(self):
        if self.mode == 'ae':
            self._set_gradients(train_gen=True, train_disc=False, train_class=False)
            return

        # Determine phase for AAE/Classif modes
        is_gen_phase = self.gen_train
        
        if self.mode == 'aae':
            self._set_gradients(train_gen=is_gen_phase, train_disc=not is_gen_phase, train_class=False)
        elif self.mode == 'classif':
            self._set_gradients(train_gen=is_gen_phase, train_disc=not is_gen_phase, train_class=True)

    def after_batch(self):
        # Switch phases logic
        if self.mode in ['aae', 'classif']:
            self.count += 1
            if self.count % self.switch_freq == 0:
                self.gen_train = not self.gen_train

    def _set_gradients(self, train_gen, train_disc, train_class):
        # Access the raw model (unwrap from DistributedDataParallel if needed)
        model = self.learn.model
        
        # Freeze/Unfreeze Discriminator
        for name, param in model.discriminator.named_parameters():
            param.requires_grad = train_disc
            
        # Freeze/Unfreeze Encoder/Decoder (Generator)
        # Note: In 'classif' mode, we might want encoder trainable with classifier
        for name, param in model.encoder.named_parameters():
            param.requires_grad = train_gen or train_class
        for name, param in model.decoder.named_parameters():
            param.requires_grad = train_gen
            
        # Freeze/Unfreeze Classifier
        for name, param in model.classifier.named_parameters():
            param.requires_grad = train_class

# --- 2. Loss Function ---

class AAELoss:
    def __init__(self, mode='ae'):
        self.mode = mode
        self.huber = nn.HuberLoss(delta=0.5)
        self.bce = nn.BCELoss() # For discriminator
        self.ce = nn.CrossEntropyLoss() # For classifier (multi-class) or BCEWithLogits (binary)

    def __call__(self, output, target):
        # Unpack dictionary from model
        recon = output['reconstruction']
        inp = output['input']
        disc_fake = output['disc_fake']
        disc_real = output['disc_real']
        logits = output['logits']
        
        # 1. Reconstruction Loss
        recon_loss = self.huber(recon, inp)
        
        if self.mode == 'ae':
            return recon_loss

        # 2. Adversarial Loss
        # We need to know if we are currently updating Generator or Discriminator.
        # We can infer this from the requires_grad state of the discriminator.
        # Note: output['disc_fake'] has grad_fn if Generator is training (to fool disc), 
        # or if Discriminator is training (to detect fake).
        
        # Hacky but effective way to detect phase from model state without passing the callback
        # Ideally, use the Callback to store the partial losses in `learn.loss_grad`
        is_disc_training = False
        for param in output['disc_fake'].grad_fn.next_functions[0][0].next_functions:
            # tracing back to see if discriminator weights are involved
            pass 
        # Easier check: look at a specific parameter's requires_grad that we know we toggle
        # However, for pure loss calculation, we calculate both and weights handle the update.
        
        # Generator Loss: Fool the discriminator (Fake should be 1)
        adv_gen_loss = self.bce(disc_fake, torch.ones_like(disc_fake))
        
        # Discriminator Loss: Real is 1, Fake is 0
        adv_disc_real = self.bce(disc_real, torch.ones_like(disc_real))
        adv_disc_fake = self.bce(disc_fake, torch.zeros_like(disc_fake))
        adv_disc_loss = 0.5 * (adv_disc_real + adv_disc_fake)

        # 3. Classification Loss
        # Target comes from DataLoaders. 
        # If using CategoryBlock, target is an int index.
        class_loss = self.ce(logits, target)

        if self.mode == 'aae':
            # In AAE mode, we return the sum, but Gradients are controlled by Callback.
            # If gradients are frozen, that part of the loss won't update weights.
            return recon_loss + 0.1 * (adv_gen_loss + adv_disc_loss)
        
        if self.mode == 'classif':
            return 0.1*recon_loss + 0.1*(adv_gen_loss + adv_disc_loss) + 0.8*class_loss

# --- 3. Metrics & Visualization ---

class ExtractLatent(Callback):
    "Extracts latent vectors during validation"
    def before_validate(self):
        self.preds = []
        self.targs = []
    
    def after_batch(self):
        output = self.learn.pred
        self.preds.append(output['zi'].detach().cpu())
        self.targs.append(self.learn.y.detach().cpu())
        
    def after_validate(self):
        self.learn.latent_preds = torch.cat(self.preds)
        self.learn.latent_targs = torch.cat(self.targs)

# --- 4. Math/Stats Utils (Kept mostly original) ---

def distrib_regul_regression(z, target, nbins: int=100, get_reg: bool=False):
    # Ensure inputs are numpy
    if isinstance(target, torch.Tensor): target = target.numpy()
    if isinstance(z, torch.Tensor): z = z.numpy()
        
    bin_edges = np.linspace(target.min(), target.max(), nbins+1)
    labels = np.digitize(target, bin_edges)
    bin_index_per_label = [int(label) for label in labels]

    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]

    reg = LinearRegression().fit(z, target.flatten(), sample_weight=weights)
    out = np.dot(z, reg.coef_) + reg.intercept_

    return (out, reg) if get_reg else out

def compute_main_direction(predictions_embedded, safelab):
    # Simplified robustness
    dark_mask = safelab == 0
    light_mask = safelab == 1
    
    dark_mean = np.mean(predictions_embedded[dark_mask, :], axis=0)
    light_mean = np.mean(predictions_embedded[light_mask, :], axis=0)
    
    diff = light_mean - dark_mean
    
    # Avoid div by zero
    if diff[0] == 0: m = 0 
    else: m = diff[1] / diff[0]
    
    b = dark_mean[1] - m * dark_mean[0]
    
    # Simple logic for plotting line segment
    x_vals = predictions_embedded[:, 0]
    min_x, max_x = x_vals.min(), x_vals.max()
    
    start = (min_x, min_x * m + b)
    end = (max_x, max_x * m + b)
    
    return start, end

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        from scipy.signal import triang
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window