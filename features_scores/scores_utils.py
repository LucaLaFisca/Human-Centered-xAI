import torch
import torch.fft
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as tfms
from tqdm.auto import tqdm

def compute_fft_scores(image_paths, radius=30):
    """
    Calcule le ratio d'énergie haute fréquence pour une liste d'images.
    Retourne un dictionnaire { 'nom_image.jpg': score_hf }.
    En cas d'erreur sur une image, lui assigne le score de l'image précédente.
    """
    scores = {}
    
    # VARIABLE D'ÉTAT : Mémorise le dernier score calculé avec succès.
    # Initialisée à 0.0 pour couvrir le cas où la toute première image serait corrompue.
    last_valid_score = 0.0 
    
    for img_path in tqdm(image_paths, desc="Calcul des scores FFT"):
        try:
            # 1. Chargement et conversion en niveaux de gris
            img = read_image(str(img_path), mode=ImageReadMode.GRAY)

            # SECURITE 1 : Si le mode GRAY a échoué et que l'image a plus d'un canal
            if img.shape[0] > 1:
                # On ne prend que les 3 premiers canaux (ignore l'alpha) et on force en gris
                img = tfms.rgb_to_grayscale(img[:3])

            img = tfms.resize(img, [200, 200], antialias=True).squeeze()

            # 2. Passage dans le domaine fréquentiel et centrage
            fshift = torch.fft.fftshift(torch.fft.fft2(img))
            magnitude_spectrum = torch.abs(fshift)

            # 3. Création du masque de filtrage (basses fréquences)
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            y = torch.arange(-crow, rows - crow).view(-1, 1)
            x = torch.arange(-ccol, cols - ccol).view(1, -1)
            mask = (x**2 + y**2 <= radius**2)

            # 4. Bilan énergétique
            total_energy = torch.sum(magnitude_spectrum)
            lf_energy = torch.sum(magnitude_spectrum[mask])
            hf_energy = total_energy - lf_energy

            # 5. Calcul et stockage du ratio
            hf_ratio = (hf_energy / total_energy).item() if total_energy > 0 else 0
            scores[img_path.name] = hf_ratio
            
            # MISE À JOUR DE LA MÉMOIRE : 
            # Si on arrive ici, c'est que tout a fonctionné.
            # On écrase l'ancienne valeur par ce nouveau score valide.
            last_valid_score = hf_ratio
            
        except Exception as e:
            # IMPUTATION EN CAS DE CRASH :
            # L'image a déclenché une erreur (ex: read_image a échoué ou shape n'est pas 2D).
            # On utilise la mémoire (last_valid_score) pour ne pas laisser de trou dans le dictionnaire.
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
