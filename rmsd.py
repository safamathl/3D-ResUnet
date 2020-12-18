import numpy as np
from rmsd.calculate_rmsd import rmsd as rm


import SimpleITK as sitk


	
y_true = sitk.ReadImage("/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/liver_pred/pred1.nii")
y_pred = sitk.ReadImage("/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/seg/segmentation1.nii")
#t = np.array(y_true)
#p = np.array(y_pred) 

rsmd = rm(y_true, y_pred)

print("RSMD" , rsmd)
