import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the .nii.gz file
img = nib.load('test/PET.nii.gz')

# Get the image data as a numpy array
data = img.get_fdata()

# Get the affine transformation matrix
affine = img.affine

# Get the header information
header = img.header

# Show a cut of the image in matplotlib
import matplotlib.pyplot as plt

# Display a coronal slice of the image

slice_idx = data.shape[1] // 2
plt.imshow(data[:, slice_idx, :].T, cmap='gray', origin='lower', vmax=10000)
plt.colorbar()
plt.title(f'Coronal slice at index {slice_idx}')
plt.show()