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
# Command used to parcellate:
#    wb_command -cifti-parcellate sub-NDARAD481FXF_1_task-task-rest_space-fsLR_den-32k_bold-dtseries.nii \
#    Schaefer2018_200Parcels_17Networks_order.dlabel.nii COLUMN output.ptseries.nii
#
# Extending CPAC
#
# Prior probability maps added in the file func_preproc_modified.py
# 
# I couldn't figure out how to integrate the time series extraction tool. :(

