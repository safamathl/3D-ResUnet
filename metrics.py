"""
Test script
"""

import os
import copy
import collections
from time import time

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology

from net.ResUNet import ResUNet
from utilities.calculate_metrics import Metirc

import parameter as para

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# In order to calculate the two variables defined by dice_globaldice_intersection = 0.0  
dice_union = 0.0

file_name = []  # file name
time_pre_case = []  # Singleton data consumption time

# Define evaluation indicators
liver_score = collections.OrderedDict()
liver_score['dice'] = []
liver_score['jacard'] = []
liver_score['voe'] = []
liver_score['fnr'] = []
liver_score['fpr'] = []
liver_score['assd'] = []
liver_score['rmsd'] = []
liver_score['msd'] = []


for file_index, file in enumerate(os.listdir(para.test_ct_path)):
    # Read the gold standard into memory
    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    # Extract the largest connected domain of the liver, remove small areas, and fill the internal holes
    pred = sitk.ReadImage(os.path.join(para.pred_path, file.replace('volume', 'pred')), sitk.sitkUInt8)
    liver_seg = sitk.GetArrayFromImage(pred)
    liver_seg[liver_seg > 0] = 1
	
	
	
    # Calculate segmentation evaluation index
    liver_metric = Metirc(seg_array, liver_seg, ct.GetSpacing())

    liver_score['dice'].append(liver_metric.get_dice_coefficient()[0])
    liver_score['jacard'].append(liver_metric.get_jaccard_index())
    liver_score['voe'].append(liver_metric.get_VOE())
    liver_score['fnr'].append(liver_metric.get_FNR())
    liver_score['fpr'].append(liver_metric.get_FPR())
    liver_score['assd'].append(liver_metric.get_ASSD())
    liver_score['rmsd'].append(liver_metric.get_RMSD())
    liver_score['msd'].append(liver_metric.get_MSD())

    dice_intersection += liver_metric.get_dice_coefficient()[1]
    dice_union += liver_metric.get_dice_coefficient()[2]

    
# Write evaluation indicators into exel
liver_data = pd.DataFrame(liver_score, index=file_name)
liver_data['time'] = time_pre_case

liver_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(liver_data.columns))
liver_statistics.loc['mean'] = liver_data.mean()
liver_statistics.loc['std'] = liver_data.std()
liver_statistics.loc['min'] = liver_data.min()
liver_statistics.loc['max'] = liver_data.max()

writer = pd.ExcelWriter('./result.xlsx')
liver_data.to_excel(writer, 'liver')
liver_statistics.to_excel(writer, 'liver_statistics')
writer.save()

# dice global
print('dice global:', dice_intersection / dice_union)








































