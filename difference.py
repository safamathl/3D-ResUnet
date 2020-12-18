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
print("hello")
import cv2
import sys



y_true = sitk.ReadImage("/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/liver_pred/pred1.nii")
y_pred = sitk.ReadImage("/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/seg/segmentation1.nii")
print("img true")
print (y_true)
print("img predict")
print (y_pred)
print("hello2")
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_true, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_true, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true, y_pred)
print('F1 score: %f' % f1)
 # confusion matrix
matrix = confusion_matrix(y_true, y_pred)
print(matrix)