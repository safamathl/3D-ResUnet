import math

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology


class Metirc():
    
    def __init__(self, real_mask, pred_mask, voxel_spacing):
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing

        self.real_mask_surface_pts = self.get_surface(real_mask, voxel_spacing)
        self.pred_mask_surface_pts = self.get_surface(pred_mask, voxel_spacing)

        self.real2pred_nn = self.get_real2pred_nn()
        self.pred2real_nn = self.get_pred2real_nn()

    def get_surface(self, mask, voxel_spacing):


        kernel = morphology.generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask

        surface_pts = surface.nonzero()

        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))

        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def get_pred2real_nn(self):


        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)

        return nn

    def get_real2pred_nn(self):

        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)

        return nn

    # Dice
    def get_dice_coefficient(self):

        intersection = (self.real_mask * self.pred_mask).sum()
        union = self.real_mask.sum() + self.pred_mask.sum()

        return 2 * intersection / union, 2 * intersection, union

    def get_jaccard_index(self):

        intersection = (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return intersection / union

    def get_VOE(self):


        return 1 - self.get_jaccard_index()

    def get_RVD(self):

        return float(self.pred_mask.sum() - self.real_mask.sum()) / float(self.real_mask.sum())

    def get_FNR(self):

        fn = self.real_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fn / union

    def get_FPR(self):

        fp = self.pred_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fp / union


    def get_ASSD(self):

        return (self.pred2real_nn.sum() + self.real2pred_nn.sum()) / \
               (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])

    def get_RMSD(self):

        return math.sqrt((np.power(self.pred2real_nn, 2).sum() + np.power(self.real2pred_nn, 2).sum()) /
                         (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0]))

    def get_MSD(self):

        return max(self.pred2real_nn.max(), self.real2pred_nn.max())
