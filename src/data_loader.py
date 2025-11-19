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

def normalise_slice(image_slice: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to the range [0, 1]."""
    min_val = np.min(image_slice)
    max_val = np.max(image_slice)

    return (
        (image_slice - min_val) / (max_val - min_val)
        if max_val - min_val > 1e-6
        else image_slice
    )


def normalise_volume(volume_patch: np.ndarray) -> np.ndarray:
    """Normalizes a 3D patch to the range [0, 1]."""
    patch = volume_patch.astype(np.float32)
    min_val, max_val = np.min(patch), np.max(patch)

    if max_val - min_val > 1e-6:
        return (patch - min_val) / (max_val - min_val)
    else:
        return patch



@dataclass
class Patient:
    """Class for keeping track of a patient."""

    id: str
    ctres_filepath: str
    ct_filepath: str
    pet_filepath: str
    suv_filepath: str
    seg_filepath: str
    has_segmentation: Optional[bool] = None

    def get_input_filepath(self, filename_key: str) -> str:
        """Helper to get specific input modality filepath based on a key."""
        key = filename_key.upper()
        if key == "PET.NII.GZ" or key == "PET.NII":
            return self.pet_filepath
        elif key == "CT.NII.GZ" or key == "CT.NII":
            return self.ct_filepath
        elif key == "CTRES.NII.GZ" or key == "CTRES.NII":
            return self.ctres_filepath
        elif key == "SEG.NII.GZ" or key == "SEG.NII":
            return self.seg_filepath
        elif key == "SUV.NII.GZ" or key == "SUV.NII":
            return self.suv_filepath
        else:
            raise ValueError(f"Unknown input filename key: {filename_key}")

    @staticmethod
    def from_folder_path(folder_path: str) -> "Patient":
        """
        Create a Patient object from a folder_path.

        Args:
            folder_path (str): folder_path containing the subject data.
        """
        # find the files in the folder_path recursively (they can be in subfolders)
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        # filter the files to find the ones we need
        ctres_filepath = [f for f in files if "CTres.nii.gz" or "CTres.nii" in f][0]
        ct_filepath = [f for f in files if "CT.nii.gz" or "CT.nii" in f][0]
        pet_filepath = [f for f in files if "PET.nii.gz" or "PET.nii" in f][0]
        suv_filepath = [f for f in files if "SEG.nii.gz" or "SEG.nii" in f][0]
        seg_filepath = [f for f in files if "SEG.nii.gz" or "SEG.nii" in f][0]

        # check if the files exist
        if not os.path.exists(ctres_filepath):
            raise FileNotFoundError(f"File {ctres_filepath} does not exist")
        if not os.path.exists(ct_filepath):
            raise FileNotFoundError(f"File {ct_filepath} does not exist")
        if not os.path.exists(pet_filepath):
            raise FileNotFoundError(f"File {pet_filepath} does not exist")
        if not os.path.exists(suv_filepath):
            raise FileNotFoundError(f"File {suv_filepath} does not exist")
        if not os.path.exists(seg_filepath):
            raise FileNotFoundError(f"File {seg_filepath} does not exist")

        # check if the segmentation file has non zero values
        seg_img = nib.load(seg_filepath)
        seg_data = seg_img.get_fdata()
        has_segmentation = True if seg_data.any() else False

        return Patient(
            id=os.path.basename(folder_path),
            ctres_filepath=ctres_filepath,
            ct_filepath=ct_filepath,
            pet_filepath=pet_filepath,
            suv_filepath=suv_filepath,
            seg_filepath=seg_filepath,
            has_segmentation=has_segmentation,
        )


def get_patients(data_folder_path: str = "data/") -> list:
    """
    Get a list of patient directories from the given data folder path.
    Args:
        data_folder_path (str): Path to the folder containing patient directories.
    Returns:
        list: List of patient directory names.
    """
    try:
        patients = [
            Patient.from_folder_path(os.path.join(data_folder_path, d))
            for d in os.listdir(data_folder_path)
            if os.path.isdir(os.path.join(data_folder_path, d))
        ]
        return patients
    except FileNotFoundError:
        print(
            f"Error: Directory not found at {data_folder_path}. Please ensure the path is correct."
        )
        return []


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
        try:
            for key in self.filename_keys:
                file_path = current_patient.get_input_filepath(key)
                volume = nib.load(file_path).get_fdata()
                img_slice = np.take(volume, indices=slice_index, axis=self.slice_axis)
                img_slice_normalized = normalise_slice(img_slice)
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
        except FileNotFoundError as e:
            print(
                f"FATAL Error (FileNotFound) for global_slice_idx {global_slice_idx} (Patient ID: {current_patient.id}, slice_in_patient: {slice_index}): {e}"
            )
            dummy_h, dummy_w = 256, 256
            return torch.zeros(
                (len(self.input_filenames_keys), dummy_h, dummy_w), dtype=torch.float32
            ), torch.zeros((1, dummy_h, dummy_w), dtype=torch.float32)
        except Exception as e:
            print(
                f"FATAL Error during __getitem__ for global_slice_idx {global_slice_idx} (Patient ID: {current_patient.id}, slice_in_patient: {slice_index}): {e}"
            )
            dummy_h, dummy_w = 256, 256
            return torch.zeros(
                (len(self.input_filenames_keys), dummy_h, dummy_w), dtype=torch.float32
            ), torch.zeros((1, dummy_h, dummy_w), dtype=torch.float32)
        

# --- Load Patients ---
patients = get_patients(data_folder_path=DATASET_PATH)

# Sélectionner un quart des patients au hasard
random.seed(RANDOM_SEED)
patients = random.sample(patients, len(patients) // 5)

# --- Dataset Init ---
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