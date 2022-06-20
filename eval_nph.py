# -*- coding: utf-8 -*-

import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial
from glob import glob
import csv
# import nibabel as nib
from ventricle_mask import VentricleMask


# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject
# resultDir = ("/home/muhan/pg17-muhan/NPH_project/Experiments/"
#             "DNN_journal/result/rudolph/ss0.1/nph/venmask")
resultDir = ("/home/muhan/pg17-muhan/NPH_project/Experiments/"
             "DNN_journal/result/dnn_15_25_noaug/nph")
testfileDir = ("/iacl/pg17/muhan/NPH_project/Experiments/DNN_journal/DNN/NPH_test")
fpre = 'nph_dnn1525_noaug'
fDice = 'dice_' + fpre + '.csv'
fVol = 'vol_' + fpre + '.csv'
fH95 = 'h95_' + fpre + '.csv'
# For example: '/data/Utrecht/0'
manualDir = '/iacl/pg17/muhan/NPH_project/Experiments/DNN_journal/DNN/NPH_data'
# For example: '/output/teamname/0'
testList = sorted(glob(os.path.join(testfileDir, '*_image.nii.gz')))
n = len(testList)


def do():
    """Main function"""
    # resultFilename = getResultFilename(participantDir)
    labelList = np.array([51, 52, 4, 11], dtype=float)
    dsc = np.zeros([n, len(labelList)])
    h95 = np.zeros([n, len(labelList)])
    subjIDlist = []
    #volMan = np.zeros([n, len(labelList)])
    volSeg = np.zeros([n, len(labelList)])

    k = n
    for i in range(k):
        testFilename = testList[i]
        subjID = os.path.basename(testFilename)
        subjID = subjID[0:5]
        resultFilename = os.path.join(resultDir, subjID + '_dnn.nii.gz')
        manualFilename = os.path.join(manualDir, subjID + '_label.nii.gz')
        manualImage, resultImage = getImages(manualFilename, resultFilename)
        manualMask = VentricleMask(segImage=manualImage)
        resultMask = VentricleMask(segImage=resultImage)
        subjIDlist.append(subjID)
        dsc[i, ] = getDSC(manualMask, resultMask, labelList)
        #volMan[i, ] = getVol(manualMask, labelList)
        volSeg[i, ] = getVol(resultMask, labelList)
        h95[i, ] = getHausdorff(manualMask, resultMask, labelList)

    subjIDlist = np.array([subjIDlist])
    # Write Dice file
    b = dsc[:k, :]
    a = np.concatenate((subjIDlist.T, b), axis=1)
    with open(fDice, 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(a)

    # Write Volume file
    b = volSeg[:k, :]
    # b = volMan[:k, :]
    a = np.concatenate((subjIDlist.T, b), axis=1)
    with open(fVol, 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(a)

    # Write Hausdorff distance file
    b = h95[:k, :]
    a = np.concatenate((subjIDlist.T, b), axis=1)
    with open(fH95, 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(a)

    # manualnib = nib.load(manualFilename)
    # manualArraynib = manualnib.get_fdata()
    # manualArraynibRV = np.where(manualArraynib == 52, 1, 0)
    # resultnib = nib.load(resultFilename)
    # resultArraynib = resultnib.get_fdata()
    # resultArraynibRV = np.where(resultArraynib == 52, 1, 0)
    # dsc2 = (2 * np.sum(manualArraynibRV*resultArraynibRV) /
    #         (np.sum(manualArraynibRV) + np.sum(resultArraynibRV)))


    # avd = getAVD(manualImage, resultImage)
    # recall, f1 = getLesionDetection(testImage, resultImage)
    # print 'Dice', np.average(dsc, 0), '(higher is better, max=1)'
    # print('Dice', dsc2)
    # print 'Volume', vol
    # print('RV Volume', np.sum(manualArraynibRV))
    # print('Dice', dsc2, '(higher is better, max=1)')
    # print 'HD', np.average(h95, 0), 'mm',  '(lower is better, min=0)'
    # print 'AVD',                 avd,  '%',  '(lower is better, min=0)'
    # print 'Lesion detection', recall,       '(higher is better, max=1)'
    # print 'Lesion F1',            f1,       '(higher is better, max=1)'


def getImages(manualFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    manualImage = sitk.ReadImage(manualFilename)
    resultImage = sitk.ReadImage(resultFilename)

    # Check for equality
    assert manualImage.GetSize() == resultImage.GetSize()

    return manualImage, resultImage

    # # Get meta data from the test-image, needed for some sitk methods to check this
    # resultImage.CopyInformation(testImage)

    # # Remove non-WMH from the test and result images, since we don't evaluate on that
    # maskedTestImage = sitk.BinaryThreshold(testImage, 0.5,  1.5, 1, 0) # WMH == 1
    # nonWMHImage     = sitk.BinaryThreshold(testImage, 1.5,  2.5, 0, 1) # non-WMH == 2
    # maskedResultImage = sitk.Mask(resultImage, nonWMHImage)

    # # Convert to binary mask
    # if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
    #     bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    # else:
    #     bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)

    # return maskedTestImage, bResultImage


# def getResultFilename(participantDir):
#     """Find the filename of the result image.

#     This should be result.nii.gz or result.nii. If these files are not present,
#     it tries to find the closest filename."""
#     files = os.listdir(participantDir)

#     if not files:
#         raise Exception("No results in " + participantDir)

#     resultFilename = None
#     if 'result.nii.gz' in files:
#         resultFilename = os.path.join(participantDir, 'result.nii.gz')
#     elif 'result.nii' in files:
#         resultFilename = os.path.join(participantDir, 'result.nii')
#     else:
#         # Find the filename that is closest to 'result.nii.gz'
#         maxRatio = -1
#         for f in files:
#             currentRatio = difflib.SequenceMatcher(a = f, b = 'result.nii.gz').ratio()

#             if currentRatio > maxRatio:
#                 resultFilename = os.path.join(participantDir, f)
#                 maxRatio = currentRatio

#     return resultFilename


def getDSC(manualMask, resultMask, labelList):
    """Compute the Dice Similarity Coefficient."""
    # dice = np.zeros(len(labelList))
    dice = np.zeros(len(labelList))
    # vol = np.zeros(len(labelList))
    for i in range(len(labelList)):
        tempManualMask = manualMask.get_mask(labelList[i])
        tempResultMask = resultMask.get_mask(labelList[i])
        manualArray = sitk.GetArrayFromImage(tempManualMask).flatten()
        resultArray = sitk.GetArrayFromImage(tempResultMask).flatten()
        # dice[i] = 1.0 - scipy.spatial.distance.dice(manualArray, resultArray)
        dice[i] = 2 * np.sum(manualArray*resultArray) / (np.sum(manualArray) +
                                                          np.sum(resultArray))

    return dice


def getVol(resultMask, labelList):
    """Compute the mask volume."""

    vol = np.zeros(len(labelList))
    for i in range(len(labelList)):
        tempResultMask = resultMask.get_mask(labelList[i])
        resultArray = sitk.GetArrayFromImage(tempResultMask).flatten()
        vol[i] = np.sum(resultArray)

    return vol


def getHausdorff(manualMask, resultMask, labelList):
    """Compute the Hausdorff distance."""
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    h95 = np.zeros(len(labelList))

    for i in range(len(labelList)):
        tempManualMask = manualMask.get_mask(labelList[i])
        tempResultMask = resultMask.get_mask(labelList[i])
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


def getAVD(testImage, resultImage):
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)

    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) \
        / float(testStatistics.GetSum()) * 100


if __name__ == "__main__":
    do()
