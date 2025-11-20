import os
import torch
import nibabel as nib
import numpy as np
from src.model import UNet

# Configuration
MODEL_PATH = "models/unet_model_1.pth"  # Chemin vers votre modèle
PET_FILE = "test/PET.nii.gz"  # Fichier PET
CT_FILE = "test/CT.nii.gz"  # Fichier CT
OUTPUT_FILE = "test/PET_segmentation.nii.gz"  # Fichier de sortie
SLICE_AXIS = 2  # Axe de découpage (doit correspondre à l'entraînement)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_slice(img_slice):
    """Normalise une slice entre 0 et 1"""
    if np.max(img_slice) - np.min(img_slice) > 1e-6:
        return (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
    return img_slice

def infer_volume(model, volume, slice_axis, device):
    """
    Effectue l'inférence sur un volume 3D slice par slice
    
    Args:
        model: Modèle U-Net chargé
        volume: Volume 3D d'entrée (numpy array) - peut être 3D (H, W, D) ou 4D (H, W, D, C)
        slice_axis: Axe de découpage
        device: Device (CPU ou CUDA)
    
    Returns:
        Volume 3D de segmentation (numpy array)
    """
    model.eval()
    
    # Déterminer si on a plusieurs canaux (PET + CT)
    is_multichannel = len(volume.shape) == 4
    
    # Créer un volume de sortie vide
    if is_multichannel:
        output_shape = list(volume.shape[:3])  # Prendre seulement H, W, D
    else:
        output_shape = list(volume.shape)
    segmentation_volume = np.zeros(output_shape, dtype=np.float32)
    
    # Nombre de slices selon l'axe choisi
    n_slices = volume.shape[slice_axis]
    
    print(f"Traitement de {n_slices} slices...")
    print(f"Volume shape: {volume.shape}, Multi-canal: {is_multichannel}")
    
    with torch.no_grad():
        for slice_idx in range(n_slices):
            # Extraire la slice
            img_slice = np.take(volume, indices=slice_idx, axis=slice_axis)
            
            if is_multichannel:
                # Si multi-canal, normaliser chaque canal séparément
                normalized_channels = []
                for c in range(img_slice.shape[-1]):
                    channel = img_slice[..., c]
                    normalized_channels.append(normalize_slice(channel))
                img_slice_normalized = np.stack(normalized_channels, axis=0)  # Shape: (C, H, W)
                # Ajouter dimension batch: (1, C, H, W)
                input_tensor = torch.from_numpy(img_slice_normalized).float().unsqueeze(0)
            else:
                # Normaliser
                img_slice_normalized = normalize_slice(img_slice)
                # Ajouter les dimensions (batch, channel)
                # Shape: (1, 1, H, W)
                input_tensor = torch.from_numpy(img_slice_normalized).float()
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            
            input_tensor = input_tensor.to(device)
            
            # Inférence
            output_logits = model(input_tensor)
            output_probs = torch.sigmoid(output_logits)
            
            # Seuillage à 0.5 pour obtenir une segmentation binaire
            pred_slice = (output_probs > 0.5).float()
            
            # Convertir en numpy et retirer les dimensions (batch, channel)
            pred_slice = pred_slice.cpu().numpy().squeeze()
            
            # Placer la slice dans le volume de sortie
            if slice_axis == 0:
                segmentation_volume[slice_idx, :, :] = pred_slice
            elif slice_axis == 1:
                segmentation_volume[:, slice_idx, :] = pred_slice
            elif slice_axis == 2:
                segmentation_volume[:, :, slice_idx] = pred_slice
            
            if (slice_idx + 1) % 10 == 0:
                print(f"  Slice {slice_idx + 1}/{n_slices} traitée")
    
    print("Traitement terminé!")
    return segmentation_volume

def main():
    print(f"Device utilisé: {DEVICE}")
    
    # Charger les images PET et CT
    print(f"\nChargement de l'image PET depuis {PET_FILE}...")
    pet_nii = nib.load(PET_FILE)
    pet_volume = pet_nii.get_fdata()
    print(f"Shape PET: {pet_volume.shape}")
    
    print(f"Chargement de l'image CT depuis {CT_FILE}...")
    ct_nii = nib.load(CT_FILE)
    ct_volume = ct_nii.get_fdata()
    print(f"Shape CT: {ct_volume.shape}")
    
    # Vérifier que les deux volumes ont la même taille
    if pet_volume.shape != ct_volume.shape:
        raise ValueError(f"Les volumes PET et CT doivent avoir la même taille. PET: {pet_volume.shape}, CT: {ct_volume.shape}")
    
    # Combiner PET et CT en un volume multi-canal
    print("Combinaison des volumes PET et CT...")
    volume = np.stack([pet_volume, ct_volume], axis=-1)  # Shape: (H, W, D, 2)
    
    n_channels = 2
    print(f"Shape du volume combiné: {volume.shape}")
    
    # Charger le modèle avec 2 canaux
    print(f"\nChargement du modèle depuis {MODEL_PATH}...")
    model = UNet(n_channels=n_channels, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Modèle chargé avec succès!")
    
    # Effectuer l'inférence
    print(f"\nDébut de l'inférence...")
    segmentation = infer_volume(model, volume, SLICE_AXIS, DEVICE)
    
    # Sauvegarder le résultat
    print(f"\nSauvegarde de la segmentation dans {OUTPUT_FILE}...")
    seg_nii = nib.Nifti1Image(segmentation.astype(np.uint8), pet_nii.affine, pet_nii.header)
    nib.save(seg_nii, OUTPUT_FILE)
    print("Segmentation sauvegardée avec succès!")
    
    # Statistiques
    total_voxels = np.prod(segmentation.shape)
    segmented_voxels = np.sum(segmentation > 0)
    percentage = (segmented_voxels / total_voxels) * 100
    print(f"\nStatistiques:")
    print(f"  - Voxels totaux: {total_voxels}")
    print(f"  - Voxels segmentés: {segmented_voxels}")
    print(f"  - Pourcentage segmenté: {percentage:.2f}%")

if __name__ == "__main__":
    main()
