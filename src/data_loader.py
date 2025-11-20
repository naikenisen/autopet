import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from tqdm import tqdm
import os
from src.config import *
from sklearn.model_selection import train_test_split
import random

class NiftDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        slice_axis=2,  # 0: x, 1: y, 2: z axis to slice the images
        verbose=False,
    ):
        super().__init__()

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.slice_axis = int(slice_axis)
        self.verbose = verbose
        self.lesion_label = 1
        self.slice_map: List[Tuple[int, int]] = []  # (file_idx, slice_idx)

        skipped_files = 0
        for file_idx, (img_path, label_path) in tqdm(
            enumerate(zip(self.image_paths, self.label_paths)), 
            desc="Loading NIfTI files", 
            unit="files",
            total=len(self.image_paths)
        ):
            try:
                img_header = nib.load(label_path).header
                n_slices = img_header.get_data_shape()[self.slice_axis]
                for slice_idx in range(n_slices):
                    self.slice_map.append((file_idx, slice_idx))
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {label_path}: {e}. Skipping.")
                skipped_files += 1

        self.total_slices = len(self.slice_map)
        if self.verbose:
            print(f"Total slices across all files: {self.total_slices}")
            if skipped_files > 0:
                print(f"Skipped {skipped_files} files.")

    def __len__(self) -> int:
        """Returns the total number of slices across all files."""
        return self.total_slices

    def __getitem__(self, global_slice_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the input and target tensors for a given slice index."""
        file_idx, slice_idx = self.slice_map[global_slice_idx]
        
        # Load image volume and extract slice
        image_volume = nib.load(self.image_paths[file_idx]).get_fdata()
        img_slice = np.take(image_volume, indices=slice_idx, axis=self.slice_axis)
        
        # Normalize image slice
        if np.max(img_slice) - np.min(img_slice) > 1e-6:
            img_slice_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
        else:
            img_slice_normalized = img_slice
        
        # Add channel dimension: (H, W) -> (1, H, W)
        input_array = np.expand_dims(img_slice_normalized, axis=0)

        # Load target segmentation slice
        label_volume = nib.load(self.label_paths[file_idx]).get_fdata()
        label_slice = np.take(label_volume, indices=slice_idx, axis=self.slice_axis)

        # Binary mask for the target
        binary_target_array = (label_slice == self.lesion_label).astype(np.float32)
        binary_target_array = np.expand_dims(binary_target_array, axis=0)

        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(binary_target_array).float()

        return input_tensor, target_tensor


# Get all image and label file paths
images_dir = os.path.join(DATASET_PATH, "images")
labels_dir = os.path.join(DATASET_PATH, "labels")

# Get all image files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.nii', '.nii.gz'))])

# Create corresponding paths
image_paths = [os.path.join(images_dir, f) for f in image_files]
label_paths = [os.path.join(labels_dir, f) for f in image_files]

# Subsample 1/5 of the data
random.seed(RANDOM_SEED)
indices = random.sample(range(len(image_paths)), len(image_paths) // 5)
image_paths = [image_paths[i] for i in indices]
label_paths = [label_paths[i] for i in indices]

# Split into train and validation
train_img, val_img, train_lbl, val_lbl = train_test_split(
    image_paths, label_paths, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
)

train_dataset = NiftDataset(
    image_paths=train_img,
    label_paths=train_lbl,
    slice_axis=SLICE_AXIS,
)
val_dataset = NiftDataset(
    image_paths=val_img,
    label_paths=val_lbl,
    slice_axis=SLICE_AXIS,
)