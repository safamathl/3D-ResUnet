import SimpleITK as sitk
from medpy.metric import binary as bn 
import numpy as np





y_true = sitk.ReadImage("/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/liver_pred/pred1.nii")
y_pred = sitk.ReadImage("/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/seg/segmentation1.nii")

t = np.array(y_true)
p = np.array(y_pred) 

asd = bn. assd(t, p, voxelspacing=None, connectivity=1)

print("ASD" , asd)
