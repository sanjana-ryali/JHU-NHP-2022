# -*- coding: utf-8 -*-


import SimpleITK as sitk


class VentricleMask:

    def __init__(self, segImage):
        self.segImage = segImage

    def get_mask(self, label):
        mask = sitk.BinaryThreshold(self.segImage, label, label, 1, 0)
        return mask

    def get_whole_mask(self):
        mask = sitk.BinaryThreshold(self.segImage, 3.5, 55.5, 1, 0)
        return mask

    # def get_rv_mask(self):
    #     manmask = sitk.BinaryThreshold(self.manImage, 50.5, 51.5, 1, 0)
    #     segmask = sitk.BinaryThreshold(self.segImage, 50.5, 51.5, 1, 0)
    #     return manmask, segmask

    # def get_lv_mask(self):
    #     manmask = sitk.BinaryThreshold(self.manImage, 51.5, 52.5, 1, 0)
    #     segmask = sitk.BinaryThreshold(self.segImage, 51.5, 52.5, 1, 0)
    #     return manmask, segmask

    # def get_3rd_mask(self):
    #     manmask = sitk.BinaryThreshold(self.manImage, 3.5, 4.5, 1, 0)
    #     segmask = sitk.BinaryThreshold(self.segImage, 3.5, 4.5, 1, 0)
    #     return manmask, segmask

    # def get_4th_mask(self):
    #     manmask = sitk.BinaryThreshold(self.manImage, 10.5, 12.5, 1, 0)
    #     segmask = sitk.BinaryThreshold(self.segImage, 10.5, 12.5, 1, 0)
    #     return manmask, segmask
