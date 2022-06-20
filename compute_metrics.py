# -*- coding: utf-8 -*-

import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial
from glob import glob
import csv
from ventricle_mask import VentricleMask


def getImages(manualFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    manualImage = sitk.ReadImage(manualFilename)
    resultImage = sitk.ReadImage(resultFilename)

    # Check for equality
    assert manualImage.GetSize() == resultImage.GetSize()

    return manualImage, resultImage


def getDSC(manualMask, resultMask, labelList):
    """Compute the Dice Similarity Coefficient."""
    L = len(labelList) + 1
    dice = np.zeros(L)
    # vol = np.zeros(len(labelList))
    for i in range(L):
        if i < len(labelList):
            tempManualMask = manualMask.get_mask(labelList[i])
            tempResultMask = resultMask.get_mask(labelList[i])
        else:
            tempManualMask = manualMask.get_whole_mask()
            tempResultMask = resultMask.get_whole_mask()
        manualArray = sitk.GetArrayFromImage(tempManualMask).flatten()
        resultArray = sitk.GetArrayFromImage(tempResultMask).flatten()
        # dice[i] = 1.0 - scipy.spatial.distance.dice(manualArray, resultArray)
        dice[i] = 2 * np.sum(manualArray*resultArray) / (np.sum(manualArray) +
                                                          np.sum(resultArray))

    return dice


def getVol(resultMask, labelList):
    """Compute the mask volume."""
    L = len(labelList) + 1
    vol = np.zeros(L)
    for i in range(L):
        if i < len(labelList):
            tempResultMask = resultMask.get_mask(labelList[i])
        else:
            tempResultMask = resultMask.get_whole_mask()
        resultArray = sitk.GetArrayFromImage(tempResultMask).flatten()
        vol[i] = np.sum(resultArray)

    return vol


def getHausdorff(manualMask, resultMask, labelList):
    """Compute the Hausdorff distance."""
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    L = len(labelList) + 1
    h95 = np.zeros(L)

    for i in range(L):
        if i < len(labelList):
            tempManualMask = manualMask.get_mask(labelList[i])
            tempResultMask = resultMask.get_mask(labelList[i])
        else:
            tempManualMask = manualMask.get_whole_mask()
            tempResultMask = resultMask.get_whole_mask()
        # Hausdorff distance is only defined when something is detected
        resultStatistics = sitk.StatisticsImageFilter()
        resultStatistics.Execute(tempResultMask)
        if resultStatistics.GetSum() == 0:
            h95[i] = float('nan')
        else:
            # Edge detection is done by ORIGINAL - ERODED, keeping the outer
            # boundaries of lesi]ons. Erosion is performed in 2D
            eTestImage = sitk.BinaryErode(tempManualMask, (1, 1, 0))
            eResultImage = sitk.BinaryErode(tempResultMask, (1, 1, 0))

            hTestImage = sitk.Subtract(tempManualMask, eTestImage)
            hResultImage = sitk.Subtract(tempResultMask, eResultImage)

            hTestArray = sitk.GetArrayFromImage(hTestImage)
            hResultArray = sitk.GetArrayFromImage(hResultImage)
            # Convert voxel location to world coordinates. Use the coordinate
            # system of the test image
            # np.nonzero   = elements of the boundary in numpy order (zyx)
            # np.flipud    = elements in xyz order
            # np.transpose = create tuples (x,y,z)
            # manualMask.TransformIndexToPhysicalPoint converts (xyz) to world
            # coordinates (in mm)
            transferFun = tempManualMask.TransformIndexToPhysicalPoint
            transTest = np.transpose(np.flipud(np.nonzero(hTestArray))).astype(int)
            transResult = np.transpose(np.flipud(np.nonzero(hResultArray))).astype(int)
            testCoordinates = np.apply_along_axis(transferFun, 1, transTest)
            resultCoordinates = np.apply_along_axis(transferFun, 1, transResult)
            # Compute distances from test to result; and result to test
            dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
            dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
            h95[i] = max(np.percentile(dTestToResult, 95),
                         np.percentile(dResultToTest, 95))

    return h95


def getAVD(manualMask, resultMask, labelList):
    """Volume statistics."""
    
    L = len(labelList) + 1
    avd = np.zeros(L)
    for i in range(L):
        if i < len(labelList):
            tempManualMask = manualMask.get_mask(labelList[i])
            tempResultMask = resultMask.get_mask(labelList[i])
        else:
            tempManualMask = manualMask.get_whole_mask()
            tempResultMask = resultMask.get_whole_mask()
        # Compute statistics of both images
        manualStatistics = sitk.StatisticsImageFilter()
        resultStatistics = sitk.StatisticsImageFilter()
        manualStatistics.Execute(tempManualMask)
        resultStatistics.Execute(tempResultMask)       

        avd[i] = float(abs(resultStatistics.GetSum() - manualStatistics.GetSum())) \
        / float(manualStatistics.GetSum()) * 100

    return avd


def getLesionDetection(testImage, resultImage):
    """Lesion detection metrics, both recall and F1."""

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, f1
