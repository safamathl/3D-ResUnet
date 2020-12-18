# 3D knee segmengtation using deep learning

This is a repository containing code of 3D Deep  Residual-Unet Networks for Volumetric knee cartilage Segmentation.

![test result](./figures/res.png)

## Dataset

3D data of knee are used with ```255 * 255 * 216``` size and a manually created 3D segmentation label controled by experts.

> All the used data are available under the _train_ directory.

Training-data are divided into two sub-directories:

* __ct__: contains the input images.

* __seg__: contains the output images (ground truth).

## Getting Started

### Requirements

```
python 3.6
SimpleITK
numpy
pandas
scipy
```

### Step 1: Preparing dataset 

```bash
python get_training_set.py
```

### Step 2: Train ResUnet

```bash
python train.py
```

### Step 3: Test ResUnet

```bash
python test.py
```

> I write all the parameter in **parameter.py**, so first set dataset path of your own. 
after the model is well trained, run **test.py** to test the model on test set, if you want to use DenseCRF as post processing, run **./Densecrf/3D-CRF.py** if you get enough memory, or run **./Densecrf/2D-CRF.py** other wise.
