import nibabel as nib
import numpy as np
import nilearn
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# Voxelwise time-series data -> voxelwise connectomes

img = nib.load("sub-NDARAD481FXF_1_task-task-rest_desc-preproc_bold.nii")
data = nilearn.masking.apply_mask([img], 
    "sub-NDARAD481FXF_1_task-task-rest_space-bold_desc-brain_mask.nii", dtype='f')
corr = np.corrcoef(data)

print("Correlation matrix ", corr)

# Voxelwise time-series data -> parcellated time-series data

masker = NiftiLabelsMasker(labels_img="CC200.nii", standardize=True, 
    memory='nilearn_cache', verbose=5)
time_series = masker.fit_transform(img)
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
print("CC200 correlation matrix ", correlation_matrix)

masker2 = NiftiLabelsMasker(labels_img="Schaefer200_space-MNI152NLin6_res-1x1x1.nii", standardize=True, 
    memory='nilearn_cache', verbose=5)
time_series2 = masker2.fit_transform(img)
correlation_measure2 = ConnectivityMeasure(kind='correlation')
correlation_matrix2 = correlation_measure2.fit_transform([time_series2])[0]
print("Schaefer correlation matrix ", correlation_matrix2)

# Vertex-wise time series data -> average values across time
#
# Command used to create reduced file:
#       wb_command -cifti-reduce sub-NDARAD481FXF_1_task-task-rest_space-fsLR_den-32k_bold-dtseries.nii 
#       MEAN output.dscalar.nii
#
# This section is commented out because I had trouble with the file type of 
#   the WB command output - I wasn't able to plug it into any of the nibabel
#   utilities. I left the workflow here to show my thought process. 
#
# img2 = nib.load("output.dscalar.nii")
# masker = NiftiLabelsMasker(labels_img="Schaefer200_space-MNI152NLin6_res-1x1x1.nii", 
#   standardize=True, memory='nilearn_cache', verbose=5)
# data = nilearn.masking.apply_mask(["output.dscalar.nii"], 
#    "Schaefer200_space-MNI152NLin6_res-1x1x1.nii", dtype='f')
#
# time_series = masker.fit_transform("output.dscalar.nii")
#
# BIDS dataset -> cpac output
#
# CPAC default pipeline output file: 
#   sub-0025429_ses-1_task-rest_run-1_space-template_desc-preproc-1_bold.nii
# CPAC modified pipeline output file: (left it zipped to differentiate)
#    sub-0025429_ses-1_task-rest_run-1_space-template_desc-preproc-1_bold.nii.gz
# 
# I also compared the output directories of the two pipelines, and the result
#    of that comparison is in diff_output.txt
#
# I did not manage to complete the Mindboggle part due to input issues in Docker, but I got a 
# good sense of how Mindboggle works!
#
# Extending CPAC
#
# Prior probability maps added in the file func_preproc_modified.py
# 
# I couldn't figure out how to integrate the time series extraction tool. :(

