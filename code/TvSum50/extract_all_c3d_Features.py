# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:07:24 2018

@author: ziyad
"""

from Extract_C3D import Extract_C3D
import glob
import numpy as np

fileNames = glob.glob('../../Webscope_I4/ydata-tvsum50-v1_1/video/*.mp4')
#matFileNames = glob.glob('../SumMe/GT/*')
fileNames.sort()
#matFileNames.sort()

#videoFile = '../SumMe/videos/Air_Force_One.mp4'
#videoMatFile = '../SumMe/GT/Air_Force_One.mat'
extract = Extract_C3D() 
def fv_target_numpy_arrays():
    for i in range(len(fileNames)):
        videoFile = fileNames[i]
#        videoMatFile = matFileNames[i]
        
        featureMatrix, targets = extract.getVideoFeatureMatrix(videoFile)
    #targets = extract.getTargets(videoMatFile, 4480)
        new_fileName = videoFile.split('/')[-1].split('.')[0]
        print(new_fileName)
    #save numpy arrays
        np.save('../../saved_numpy_arrays/TvSum50/c3d_features/' + new_fileName + '.npy', featureMatrix)
#        np.save('../saved_numpy_arrays/targets_' + new_fileName + '.npy', targets)

def target_numpy_arrays(): # If i want to save only the target .npy arrays (instead if still extracting the full c3d features)
    for i in range(len(matFileNames)):
        videoFile = fileNames[i]
        frameCount = extract.get_video_frame_count(videoFile)
        videoMatFile = matFileNames[i]
        targets = extract.getTargets(videoMatFile, frameCount)
        new_fileName = videoMatFile.split('/')[-1].split('.')[0]
    #save numpy arrays
        np.save('../saved_numpy_arrays/targets_' + new_fileName + '.npy', targets)
    
#fv_target_numpy_arrays()
