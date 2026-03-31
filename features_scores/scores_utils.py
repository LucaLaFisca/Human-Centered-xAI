import torch
import torch.fft
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as tfms
from tqdm.auto import tqdm
import pandas as pd

def mapping(learn):
    """This function performs a comprehensive mapping of the model's predictions back to the original dataset instances,
    ensuring a strict alignment between the order of predictions and the original file paths.
    
    It ensures that we obtain the paths in the specific order of the model's predictions, 
    which is critical for subsequent feature importance analysis.
    
    Set, Original_Dataset_Index, and Source_File_Path are stored in a DataFrame to 
    track the provenance of each instance and facilitate merging with 
    feature importance scores later.
    """
    all_inputs = []
    all_dfs = []

    # Iterate over Train (0) and Valid (1)
    for ds_idx in [0, 1]:
        
        # 1. Trigger Inference (Strictly maintain reorder=False)
        # We need the predictions in the order they are processed by the DataLoader
        inputs, preds, targets = learn.get_preds(ds_idx=ds_idx, with_input=True, reorder=False)
       
        # 2. Capture the Permutation Vector
        # Dynamic addressing using learn.dls[ds_idx]
        indices = list(learn.dls[ds_idx].get_idxs())
       
        # 3. Access the Immutable Original Sequence
        # Accessing the underlying items from the dataset
        original_files = list(learn.dls[ds_idx].dataset.items)
       
        # 4. Reconstruct the Geographical Mapping (Path alignment)
        # Align original file paths with the indices used during inference
        path_mapping = [original_files[idx] for idx in indices]
       
        # 5. Semantic Extraction of Categorical Predictions
        predicted_class_indices = preds.argmax(dim=-1).numpy()
        real_target_indices = targets.argmax(dim=-1).numpy()
       
        vocab_dict = learn.dls.vocab
        predicted_class_names = [vocab_dict[idx] for idx in predicted_class_indices]
       
        # Explicitly identify the data source (Train vs Valid)
        set_name = "Train" if ds_idx == 0 else "Valid"

        # 6. Tabular Structuring of Analytical Results
        mapping_df = pd.DataFrame({
            'Set': set_name,
            'Original_Dataset_Index': indices,
            'Source_File_Path': path_mapping,
            # 'Real_Target_ID': real_target_indices,
            # 'Predicted_Class_ID': predicted_class_indices,
            # 'Predicted_Class_Name': predicted_class_names,
            # 'Correct_Prediction': (real_target_indices == predicted_class_indices)
        })

        # Temporarily store results from current iteration
        all_inputs.append(inputs)
        all_dfs.append(mapping_df)
   
    # 7. Structural Fusion (Concatenation)
    # Stack PyTorch tensors on dimension 0 (instance dimension)
    merged_inputs = torch.cat(all_inputs, dim=0)
    
    # Concatenate DataFrames with index reset to guarantee strict alignment:
    # merged_inputs[i] will correspond to merged_df.loc[i]
    merged_df = pd.concat(all_dfs, ignore_index=True)

    return merged_inputs, merged_df


def compute_fft_scores(image_paths, radius=30):
    """
    Compute the energy ratio of high frequencies for a list of images.
    Returns a list of scores.
    In case of an error on an image, assigns the score of the previous image. (NEED TO BE MODIFIED TO BE MORE ROBUST)
    """

    scores = []
    
    last_valid_score = 0.0 
    
    # tqdm for displaying progress bar
    for img_path in tqdm(image_paths, desc="Calcul des scores FFT"):
        try:

            # Read image in grayscale
            img = read_image(str(img_path), mode=ImageReadMode.GRAY)

            # Debug in case of shape issues (alpha channel, etc.)
            if img.shape[0] > 1:
                # On ne prend que les 3 premiers canaux (ignore l'alpha) et on force en gris
                img = tfms.rgb_to_grayscale(img[:3])

            img = tfms.resize(img, [200, 200], antialias=True).squeeze()

            # Application of FFT
            fshift = torch.fft.fftshift(torch.fft.fft2(img))
            magnitude_spectrum = torch.abs(fshift)

            # Mask to separate low and high frequencies
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            y = torch.arange(-crow, rows - crow).view(-1, 1)
            x = torch.arange(-ccol, cols - ccol).view(1, -1)
            mask = (x**2 + y**2 <= radius**2)

            # Extraction of energy values
            total_energy = torch.sum(magnitude_spectrum)
            lf_energy = torch.sum(magnitude_spectrum[mask])
            hf_energy = total_energy - lf_energy

            # Evaluating the score (ratio of high frequency energy)
            hf_ratio = (hf_energy / total_energy).item() if total_energy > 0 else 0
            scores.append(hf_ratio)
            
            # Updating the last valid score
            last_valid_score = hf_ratio
            
        except Exception as e:
            # In case of error (e.g., read_image fails or shape is not 2D), assign the last valid score to the current image
            print(f"Error for {img_path.name}: {e} -> Assigning previous score: {last_valid_score:.4f}")
            scores.append(last_valid_score)
            
    return scores


def find_otsu_threshold(im_gray):
    """
    WARNING : only works for 2 colors images to separate foreground and background.
    """
    # 1. Calcul de l'histogramme normalisé (probabilités)
    hist = torch.histc(im_gray, bins=256, min=0, max=255)
    p = hist / hist.sum()

    # Valeurs d'intensités [0, 1, ..., 255]
    intensites = torch.arange(256).float()

    max_sigma_b = 0
    seuil_optimal = 0

    # 2. On teste tous les seuils possibles T
    for T in range(1, 255):
        # Poids (ω) des deux classes
        w0 = p[:T].sum()
        w1 = p[T:].sum()

        if w0 == 0 or w1 == 0: continue

        # Moyennes (μ) des deux classes
        mu0 = (intensites[:T] * p[:T]).sum() / w0
        mu1 = (intensites[T:] * p[T:]).sum() / w1

        # 3. Calcul de la variance inter-classe σ²_b
        # C'est la valeur qu'on cherche à maximiser
        sigma_b = w0 * w1 * (mu0 - mu1)**2

        if sigma_b > max_sigma_b:
            max_sigma_b = sigma_b
            seuil_optimal = T

    return seuil_optimal

