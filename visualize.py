import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the PET image
img = nib.load('test/PET.nii.gz')
data = img.get_fdata()

# Load the segmentation mask
seg_img = nib.load('test/SEG.nii.gz')
seg_data = seg_img.get_fdata()

# Display a coronal slice of the image with segmentation overlay
slice_idx = data.shape[1] // 2

# Extract slices
pet_slice = data[:, slice_idx, :].T
seg_slice = seg_data[:, slice_idx, :].T

# Create figure
plt.figure(figsize=(7, 10))

# Display PET image in grayscale
plt.imshow(pet_slice, cmap='gray', origin='lower', vmax=10000)

# Overlay segmentation in red (only where seg > 0)
seg_masked = np.ma.masked_where(seg_slice == 0, seg_slice)
plt.imshow(seg_masked, cmap='Reds', origin='lower', alpha=0.5)

plt.colorbar(label='Segmentation Mask')
plt.title(f'Coronal slice at index {slice_idx} with segmentation overlay')
plt.show()