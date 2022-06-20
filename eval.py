#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse 

parser = argparse.ArgumentParser(description='Compute DSC, H95 and AVD')
parser.add_argument('-d', '--result-dir', help='Directory of the segmentation results')
parser.add_argument('-o', '--output-dir', help='Directory to write evaluation files')
parser.add_argument('-p', '--file-pre')
parser.add_argument('-t', '--type', help='nph or nmm')
parser.add_argument('--dice', type=bool, required=False, default=True)
parser.add_argument('--h95', type=bool, required=False, default=True)
parser.add_argument('--avd', type=bool, required=False, default=True)
parser.add_argument('--vol', type=bool, required=False, default=True)
args = parser.parse_args()

import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial
from glob import glob
import csv
# import nibabel as nib
from ventricle_mask import VentricleMask
from compute_metrics import *


# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject
# resultDir = ("/home/muhan/pg17-muhan/NPH_project/Experiments/"
#             "DNN_journal/result/rudolph/ss0.1/nph/venmask")

resultDir = args.result_dir
outputDir = args.output_dir

fpre = args.type + '_' + args.file_pre

# For example: '/data/Utrecht/0'
# For example: '/output/teamname/0'
if args.type == 'nmm':
    testfileDir = ("/iacl/pg17/muhan/NPH_project/Experiments/DNN_journal/DNN/NMM_test")
    manualDir = ('/iacl/pg17/muhan/NPH_project/Experiments/DNN_journal/DNN/NMM_data')
elif args.type == 'nph':
    testfileDir = ("/iacl/pg17/muhan/NPH_project/Experiments/DNN_journal/DNN/NPH_test")
    manualDir = ('/iacl/pg17/muhan/NPH_project/Experiments/DNN_journal/DNN/NPH_data')
    #testfileDir = ("/iacl/pg19/muhan/NPH_project/blitz500/manual/")
else:
    print 'Wrong type. Must be "nmm" or "nph".'

testList = sorted(glob(os.path.join(testfileDir, '*_image.nii.gz')))
n = len(testList)

labelList = np.array([51, 52, 4, 11], dtype=float)
L = len(labelList) + 1

if args.dice:
    fDice = os.path.join(outputDir, 'dice_' + fpre + '.csv')
    dsc = np.zeros([n, L])

if args.vol:
    fVol = os.path.join(outputDir, 'vol_' + fpre + '.csv')
    volSeg = np.zeros([n, L])

if args.h95:
    fH95 = os.path.join(outputDir, 'h95_' + fpre + '.csv')
    h95 = np.zeros([n, L])

if args.avd:
    fAVD = os.path.join(outputDir, 'avd_' + fpre + '.csv')
    avd = np.zeros([n, L])

subjIDlist = []
#volMan = np.zeros([n, len(labelList)])

k = n
for i in range(k):
    testFilename = testList[i]
    subjID = os.path.basename(testFilename)
    if args.type == 'nmm':
        subjID = subjID[0:4]
    elif args.type == 'nph':
        subjID = subjID[0:5]
    else:
        print 'Wrong type. Must be "nmm" or "nph".'
    resultFilename = glob(os.path.join(resultDir, subjID + '*.nii.gz'))
    # print resultDir
    resultFilename = resultFilename[0]
    manualFilename = os.path.join(manualDir, subjID + '_label.nii.gz')
    manualImage, resultImage = getImages(manualFilename, resultFilename)
    manualMask = VentricleMask(segImage=manualImage)
    resultMask = VentricleMask(segImage=resultImage)
    subjIDlist.append(subjID)

    if args.dice:
        dsc[i, ] = getDSC(manualMask, resultMask, labelList)
    #volMan[i, ] = getVol(manualMask, labelList)
    
    if args.vol:
        volSeg[i, ] = getVol(resultMask, labelList)
    
    if args.h95:
        h95[i, ] = getHausdorff(manualMask, resultMask, labelList)
    
    if args.avd:
        avd[i, ] = getAVD(manualMask, resultMask, labelList)

subjIDlist = np.array([subjIDlist])

# Write Dice file
if args.dice:
    b = dsc[:k, :]
    a = np.concatenate((subjIDlist.T, b), axis=1)
    with open(fDice, 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(a)

# Write Volume file
if args.vol:
    b = volSeg[:k, :]
# b = volMan[:k, :]
    a = np.concatenate((subjIDlist.T, b), axis=1)
    with open(fVol, 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(a)

# Write Hausdorff distance file
if args.h95:
    b = h95[:k, :]
    a = np.concatenate((subjIDlist.T, b), axis=1)
    with open(fH95, 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(a)

# Write AVD file
if args.avd:
    b = avd[:k, :]
    a = np.concatenate((subjIDlist.T, b), axis=1)
    with open(fAVD, 'wb') as outf:
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




