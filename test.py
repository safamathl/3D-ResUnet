"""
Test 
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

# Define network and load parameters
net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
net.load_state_dict(torch.load(para.module_path))
net.eval()

for file_index, file in enumerate(os.listdir(para.test_ct_path)):

    start = time()

    file_name.append(file)

    # Read CT into memory
    ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape
    
    # Truncate the gray value outside the threshold
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # min max normalization
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200

    # Interpolate CT using bicubic algorithm, the array after interpolation is still int16
    ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3)

    # Use padding for data with too few slices
    too_small = False
    if ct_array.shape[0] < para.size:
        depth = ct_array.shape[0]
        temp = np.ones((para.size, int(512 * para.down_scale), int(512 * para.down_scale))) * para.lower
        temp[0: depth] = ct_array
        ct_array = temp 
        too_small = True

    # Sliding window sampling prediction
    start_slice = 0
    end_slice = start_slice + para.size - 1
    count = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)
    probability_map = np.zeros((ct_array.shape[0], 512, 512), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            # Due to insufficient video memory, the ndarray data is directly retained here, and the calculation graph is directly destroyed after saving
            del outputs      
            
            start_slice += para.stride
            end_slice = start_slice + para.size - 1
    
        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs
        
        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (para.threshold * count)] = 1

        if too_small:
            temp = np.zeros((depth, 512, 512), dtype=np.float32)
            temp += pred_seg[0: depth]
            pred_seg = temp

    # Read the gold standard into memory
    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

# Extract the largest connected domain of the liver, remove small areas, and fill the internal holes
    pred_seg = pred_seg.astype(np.uint8)
    liver_seg = copy.deepcopy(pred_seg)
    liver_seg = measure.label(liver_seg, 4)
    props = measure.regionprops(liver_seg)
    
    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index
    
    liver_seg[liver_seg != max_index] = 0
    liver_seg[liver_seg == max_index] = 1
    
    liver_seg = liver_seg.astype(np.bool)
    morphology.remove_small_holes(liver_seg, para.maximum_hole, connectivity=2, in_place=True)
    liver_seg = liver_seg.astype(np.uint8)

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

    #dice_intersection += liver_metric.get_dice_coefficient()[1]
   # dice_union += liver_metric.get_dice_coefficient()[2]

    # Save the predicted result as nii data
    pred_seg = sitk.GetImageFromArray(liver_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.pred_path, file.replace('volume', 'pred')))

    speed = time() - start
    time_pre_case.append(speed)

    print(file_index, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')


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








































