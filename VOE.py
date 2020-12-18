import numpy as np
import SimpleITK as sitk

import itk


import SimpleITK as sitk
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection._validation import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
#from load_data import loadDataGeneral
import nibabel as nib
from keras.models import load_model
from scipy.misc import imresize
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure
import SimpleITK as sitk
#from sklearn.metrics import Dice
import cv2
import sys


#y_true = sitk.ReadImage("C:/Users/lenovo/Desktop/safa/driveknee/drive-download-20190515T061013Z-001/IBSR_13/IBSR_13_segTRI_ana.nii")
#y_pred = sitk.ReadImage("C:/Users/lenovo/Desktop/safa/driveknee/drive-download-20190515T061013Z-001/IBSR_13/IBSR_13_segTRI_predict.nii")


def voe(y_true, y_pred):
    y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return (100 * (1. - np.logical_and(y_true_f, y_pred_f).sum() / float(np.logical_or(y_true_f, y_pred_f))))
#vd = (100 * (y_true_f.sum() - y_pred_f.sum()) / float(y_pred_f.sum()))


y_true = sitk.ReadImage("IBSR_11_segTRI_ana.nii")
y_pred = sitk.ReadImage("IBSR_11_segTRI_predict.nii")

print("img true")
print (y_true)
print("img predict")

print (y_pred)


#t1 = np.array(y_true)
#p1 = np.array(y_pred) 



t1 = sitk.GetArrayViewFromImage(y_true)
p1 = sitk.GetArrayViewFromImage(y_pred)


print("hhhhhhhhiiiiiiiiiiiiiiiiihh")
print("img true_array")
print (t1)
print("img predict_array")

print (p1)

#t1 = y_true.flatten(t)
#p1 = y_pred.flatten(p)
t1 = t1.astype(int)
p1= p1.astype(int)
print("hhhhhhhhiiiiiiiiiiiiiiiiihhhhhhhhhhhhhhhhhhhhhhhh")
#intersection = np.logical_and(y_true, y_pred).sum()
#union = np.logical_or(y_true, y_pred).sum()
#(2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

#voe = 100 * (1. - intersection / union)
#vd = 100 * (((y_true).sum() - (y_pred).sum()) / (y_pred).sum())



voe = (100 * (1. - np.logical_and(t1, p1).sum() / float(np.logical_or(t1, p1))))
vd = (100 * (t1.sum() - p1.sum()) / float(p1.sum()))

#voe1 = voe(t1,p1)

print("VOE" , voe)
print("VD" , vd)