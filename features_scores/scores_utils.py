import torch
import torch.fft
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as tfms
from tqdm.auto import tqdm

def compute_fft_scores(image_paths, radius=30):
    """
    Calcule le ratio d'énergie haute fréquence pour une liste d'images.
    Retourne un dictionnaire { 'nom_image.jpg': score_hf }.
    En cas d'erreur sur une image, assignation du score de l'image précédente.
    """
    scores = {}
    
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
            scores[img_path.name] = hf_ratio
            
            # Updating the last valid score
            last_valid_score = hf_ratio
            
        except Exception as e:
            # In case of error (e.g., read_image fails or shape is not 2D), assign the last valid score to the current image
            print(f"Erreur sur {img_path.name}: {e} -> Assignation du score précédent : {last_valid_score:.4f}")
            scores[img_path.name] = last_valid_score
            
    return scores


def find_otsu_threshold(im_gray):
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

#def complexity_scores(pixels):
    # Masque pour isoler le sujet (le fond est blanc, donc < T)
    #pixels_sujet = im_gray[im_gray < T]

    # --- Feature 1 : Variance du sujet ---
    # Mesure l'étalement de la cloche "Sujet" dans l'histogramme
    #score_variance = pixels_sujet.var().item()

    # ---- ENTROPIE CALCUL ----
    #if len(pixels) == 0: return 0
    # On refait un mini-histogramme du sujet uniquement
    #h = torch.histc(pixels, bins=256, min=0, max=255)
    #p = h[h > 0] / h.sum() # On évite les log(0)
    #return -torch.sum(p * torch.log2(p)).item()
