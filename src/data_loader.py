import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
import os
from src.config import *
from sklearn.model_selection import train_test_split
import random

@dataclass
class Patient:
    """Class for keeping track of a patient."""

    id: str
    pet_filepath: str
    seg_filepath: str

    def get_input_filepath(self, filename_key):
        """Helper to get specific input modality filepath based on a key."""
        key = filename_key.upper()
        if key in ("PET.NII.GZ", "PET.NII"):
            return self.pet_filepath
        elif key in ("SEG.NII.GZ", "SEG.NII"):
            return self.seg_filepath
        else:
            raise ValueError(f"Unknown input filename key: {filename_key}")

    @staticmethod
    def from_folder_path(folder_path):
        # find the files in the folder_path recursively (they can be in subfolders)
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        # filter the files to find the ones we need
        pet_filepath = [f for f in files if "PET.nii.gz" in f or "PET.nii" in f][0]
        seg_filepath = [f for f in files if "SEG.nii.gz" in f or "SEG.nii" in f][0]

        return Patient(
            id=os.path.basename(folder_path),
            pet_filepath=pet_filepath,
            seg_filepath=seg_filepath
        )

class NiftDataset(Dataset):
    def __init__(
        self,
        patients: List[Patient],
        filenames: List[str],
        slice_axis=2,  # 0: x, 1: y, 2: z axis to slice the images in the files with.
        verbose=False,
    ):
        super().__init__()

        self.patients = patients
        self.filename_keys = [key.strip() for key in filenames]
        self.slice_axis = int(slice_axis)
        self.verbose = verbose
        self.lesion_label = 1
        self.slice_map: List[Tuple[str, str]] = []

        skipped_dirs = 0
        for patient_idx, patient_obj in tqdm(
            enumerate(self.patients), desc="Loading NIfTI files ", unit="patients"
        ):
            try:
                seg_filepath = patient_obj.seg_filepath
                img_header = nib.load(seg_filepath).header
                n_slices = img_header.get_data_shape()[self.slice_axis]
                for slice_idx_in_patient in range(n_slices):
                    self.slice_map.append((patient_idx, slice_idx_in_patient))
            except Exception as e:
                if self.verbose:
                    print(
                        f"Error loading header for patient {patient_obj.id} (SEG: {seg_filepath}): {e}. Skipping."
                    )
                skipped_dirs += 1

        self.total_slices = len(self.slice_map)
        if self.verbose:
            print(f"Total slices across all patients: {self.total_slices}")
            if skipped_dirs > 0:
                print(f"Skipped {skipped_dirs} directories.")

    def __len__(self) -> int:
        """Returns the total number of slices across all patients."""
        return self.total_slices

    def __getitem__(self, global_slice_idx: int) -> Tuple[torch.tensor, torch.Tensor]:
        """Returns the input and target tensors for a given slice index."""
        list_idx, slice_index = self.slice_map[global_slice_idx]
        current_patient: Patient = self.patients[list_idx]

        input_slices = []
        for key in self.filename_keys:
            file_path = current_patient.get_input_filepath(key)
            volume = nib.load(file_path).get_fdata()
            img_slice = np.take(volume, indices=slice_index, axis=self.slice_axis)
            if np.max(img_slice) - np.min(img_slice) > 1e-6 :
                img_slice_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
            else:
                img_slice_normalized = img_slice
            input_slices.append(img_slice_normalized)

        input_array = np.stack(input_slices, axis=0)

        # Load target segmentation slice from the Patient object
        target_volume = nib.load(current_patient.seg_filepath).get_fdata()
        target_slice = np.take(
            target_volume, indices=slice_index, axis=self.slice_axis
        )

        # Binary mask for the target
        binary_target_array = (target_slice == self.lesion_label).astype(np.float32)
        binary_target_array = np.expand_dims(
            binary_target_array, axis=0
        )  # add channel dimension: (H, W) -> (1, H, W)

        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(binary_target_array).float()

        return input_tensor, target_tensor

patients = [
    Patient.from_folder_path(os.path.join(DATASET_PATH, d))
    for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]

# Sélectionner un quart des patients au hasard
random.seed(RANDOM_SEED)
patients = random.sample(patients, len(patients) // 5)

train_patients, val_patients = train_test_split(
    patients, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
)
train_dataset = NiftDataset(
    patients=train_patients,
    filenames=INPUT_FILENAMES,
    slice_axis=SLICE_AXIS,
)
val_dataset = NiftDataset(
    patients=val_patients,
    filenames=INPUT_FILENAMES,
    slice_axis=SLICE_AXIS,
)