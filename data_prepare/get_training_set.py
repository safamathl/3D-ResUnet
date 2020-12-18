"""
Obtain a training data set that can be used to train the network

"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

import parameter as para


if os.path.exists(para.training_set_path):
    shutil.rmtree(para.training_set_path)

new_ct_path = os.path.join(para.training_set_path, 'ct')
new_seg_dir = os.path.join(para.training_set_path, 'seg')

os.mkdir(para.training_set_path)
os.mkdir(new_ct_path)
os.mkdir(new_seg_dir)

start = time()
for file in tqdm(os.listdir(para.train_ct_path)):

    #Load CT and gold standard into memory
    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(para.train_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # Fusion of the liver and liver tumor labels in the gold standard into one
    seg_array[seg_array > 0] = 1

    # Truncate the gray value outside the threshold
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # Downsample the CT data on the cross-section and resample, adjust the spacing of the z-axis of all data to 1mm
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, 1, 1), order=0)

    # Find the slices at the beginning and end of the liver area, and expand the slices outwards
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # Expand slices in both directions
    start_slice = max(0, start_slice - para.expand_slice)
    end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)

    # If the number of remaining slices is less than size at this time, just give up the data. There is very little data, so don’t worry.
    if end_slice - start_slice + 1 < para.size:
        print('!!!!!!!!!!!!!!!!')
        print(file, 'have too little slice', ct_array.shape[0])
        print('!!!!!!!!!!!!!!!!')
        continue

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    # Finally save the data as nii
    new_ct = sitk.GetImageFromArray(ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / para.down_scale), ct.GetSpacing()[1] * int(1 / para.down_scale), para.slice_thickness))

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], para.slice_thickness))

    sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
    sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))