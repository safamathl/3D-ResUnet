# -----------------------Path related parameters---------------------------------------

train_ct_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/train/ct'  #   CT data path of the original training set


train_seg_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/train/seg'  #   Original training set labeled data path

test_ct_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/ct'  #  CT data path of the original test set

test_seg_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/seg/'  #   Original test set labeled data path

training_set_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/data_prepare/train/'  #   Adresse de stockage des données utilisée pour former le réseau

pred_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/dataset/test/liver_pred'  #   Save path of network prediction results


crf_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/test/crf'  #  CRF optimization result save path


module_path = '/content/gdrive/My Drive/MICCAI-LITS2017-master/module/net550-0.251-0.231.pth'  # /  Test model path

# -----------------------Path related parameters---------------    ------------------------


# ---------------------Training data to obtain relevant parameters-----------------------------------

size = 48  # Use 48 consecutive slices as input to the network

down_scale = 0.5  # Cross-sectional downsampling factor

expand_slice = 20  # Cross-sectional downsampling factor...

slice_thickness = 1  # Normalize the spacing of all data on the z-axis to 1mm

upper, lower = 200, -200  # CT data gray cut window

# ---------------------Training data to obtain relevant parameters-----------------------------------


# -----------------------Network structure related parameters------------------------------------

drop_rate = 0.3  # dropout random drop probability

# -----------------------Network structure related parameters------------------------------------


# ---------------------Network training related parameters--------------------------------------

gpu = '0'  # The serial number of the graphics card used

Epoch = 20

learning_rate = 1e-3

learning_rate_decay = [500, 750]

alpha = 0.33  # In-depth supervision attenuation coefficient

batch_size = 1

num_workers = 3

pin_memory = True

cudnn_benchmark = True

# ---------------------Network training related parameters--------------------------------------


# ----------------------Model test related parameters-------------------------------------

threshold = 0.5  # Threshold

stride = 12  # Sliding sampling step

maximum_hole = 5e4  # Largest void area

# ----------------------Model test related parameters-------------------------------------


# ---------------------CRF post-processing optimization related parameters----------------------------------

z_expand, x_expand, y_expand = 10, 30, 30  # The number of expansions in three directions based on the predicted results

max_iter = 20  # CRF iterations

s1, s2, s3 = 1, 10, 10  # CRF Gaussian kernel parameters

# ---------------------CRF post-processing optimization related parameters----------------------------------