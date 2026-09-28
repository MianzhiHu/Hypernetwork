from nilearn import image, plotting
from pathlib import Path
import nibabel as nib

path = './ds004636/sub-s495/ses-1/func/sub-s495_ses-1_task-ANT_run-1_bold.nii.gz'

img = nib.load(path)
print(img.shape)
mean_img = image.mean_img(path)

plotting.plot_epi(
    mean_img,
    title="ANT mean BOLD"
)

plotting.show()