import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import os
from src.config import *
from sklearn.model_selection import train_test_split
import random
from scipy.ndimage import zoom

class NiftDataset(Dataset):
    def __init__(self, pet_paths, ct_paths, label_paths, slice_axis=2):
        super().__init__()
        self.pet_paths = pet_paths
        self.ct_paths = ct_paths
        self.label_paths = label_paths
        self.slice_axis = int(slice_axis)
        self.lesion_label = 1
        self.slice_map = []
        
        for file_idx, (pet_path, ct_path, label_path) in enumerate(zip(self.pet_paths, self.ct_paths, self.label_paths)):
            img_header = nib.load(label_path).header
            n_slices = img_header.get_data_shape()[self.slice_axis]
            for slice_idx in range(n_slices):
                self.slice_map.append((file_idx, slice_idx))
        
        self.total_slices = len(self.slice_map)
        print(f"Total slices: {self.total_slices}")

    def __len__(self):
        return self.total_slices

    def __getitem__(self, global_slice_idx):
        file_idx, slice_idx = self.slice_map[global_slice_idx]
        
        # Load PET volume and extract slice (channel 0)
        pet_volume = nib.load(self.pet_paths[file_idx]).get_fdata()
        pet_slice = np.take(pet_volume, indices=slice_idx, axis=self.slice_axis)
        
        # Load CT volume and extract slice (channel 1)
        ct_volume = nib.load(self.ct_paths[file_idx]).get_fdata()
        ct_slice = np.take(ct_volume, indices=slice_idx, axis=self.slice_axis)
        
        # Load target segmentation slice
        label_volume = nib.load(self.label_paths[file_idx]).get_fdata()
        label_slice = np.take(label_volume, indices=slice_idx, axis=self.slice_axis)
        
        # Resize CT and PET to match label dimensions if needed
        target_shape = label_slice.shape
        if pet_slice.shape != target_shape:
            zoom_factors = np.array(target_shape) / np.array(pet_slice.shape)
            pet_slice = zoom(pet_slice, zoom_factors, order=1)
        
        if ct_slice.shape != target_shape:
            zoom_factors = np.array(target_shape) / np.array(ct_slice.shape)
            ct_slice = zoom(ct_slice, zoom_factors, order=1)
        
        # Normalize PET slice
        if np.max(pet_slice) - np.min(pet_slice) > 1e-6:
            pet_slice_normalized = (pet_slice - np.min(pet_slice)) / (np.max(pet_slice) - np.min(pet_slice))
        else:
            pet_slice_normalized = pet_slice
        
        # Normalize CT slice
        if np.max(ct_slice) - np.min(ct_slice) > 1e-6:
            ct_slice_normalized = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice))
        else:
            ct_slice_normalized = ct_slice
        
        # Stack PET and CT as 2 channels: (2, H, W)
        input_array = np.stack([pet_slice_normalized, ct_slice_normalized], axis=0)
        
        # Binary mask for the target
        binary_target_array = (label_slice == self.lesion_label).astype(np.float32)
        binary_target_array = np.expand_dims(binary_target_array, axis=0)
        
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(binary_target_array).float()
        return input_tensor, target_tensor

# Get all image files
image_files = sorted(os.listdir(PET_IMAGE_PATH))
pet_paths = [os.path.join(PET_IMAGE_PATH, f) for f in image_files]
ct_paths = [os.path.join(CT_IMAGE_PATH, f) for f in image_files]
label_paths = [os.path.join(LABEL_PATH, f) for f in image_files]

# Subsample 1/15 of the data
random.seed(RANDOM_SEED)
indices = random.sample(range(len(pet_paths)), len(pet_paths) // 15)
pet_paths = [pet_paths[i] for i in indices]
ct_paths = [ct_paths[i] for i in indices]
label_paths = [label_paths[i] for i in indices]

# Split into train and validation
train_pet, val_pet, train_ct, val_ct, train_lbl, val_lbl = train_test_split(
    pet_paths, ct_paths, label_paths, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
)

train_dataset = NiftDataset(
    pet_paths=train_pet,
    ct_paths=train_ct,
    label_paths=train_lbl,
    slice_axis=SLICE_AXIS,
)

val_dataset = NiftDataset(
    pet_paths=val_pet,
    ct_paths=val_ct,
    label_paths=val_lbl,
    slice_axis=SLICE_AXIS,
)