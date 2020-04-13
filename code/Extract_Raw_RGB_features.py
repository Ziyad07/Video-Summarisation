# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:13:08 2018

@author: ziyad
"""

import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import scipy.io as sio

# This script has been taylored to  suit the SumMe dataset for now
#
class Extract_Raw_RGB_features(object):
        
    def vid2npy3(self, fileName):
        cap = cv2.VideoCapture(fileName)
        videoFrameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        frameCount = videoFrameCount    
        #frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        #frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frameHeight = 112
        frameWidth = 112
        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True    
        while (fc < frameCount  and ret):
            ret, frame = cap.read()
            resize = cv2.resize(frame, (frameHeight, frameWidth))
            buf[fc] = resize
            fc += 1
        cap.release()    
        return buf, frameCount
    
    def getFullVideoFeatureVector(self, videoNumpyArray, frameCount):
        feature_vector = np.zeros([37632]) # size of returned vector from c3d
        for index in tqdm(range(frameCount)):    
            video_frame = videoNumpyArray[index]
            flatten_frame = video_frame.flatten() / 255.0
            feature_vector = np.vstack((feature_vector, flatten_frame))
            
        return feature_vector
    
    def getVideoFeatureMatrix(self, fileName):
    #        fileName = 'dM06AMFLsrc.mp4' # take this out soon
        videoNumpyArray, frameCount = self.vid2npy3(fileName)
        # Remember the first line of the feature matrix is zeros which needs to be excluded
        feature_vector = self.getFullVideoFeatureVector(videoNumpyArray, frameCount)[1:] # take all except first line
        
        return feature_vector, frameCount #size (num_partitions, 4096)
        
    def readData(self, fileName):    
        video_mat_file = sio.loadmat(fileName)
        return video_mat_file
    
    def gt_mean_normalised_Targets(self, fileName):
        video_mat_file = self.readData(fileName)
        gt_score = video_mat_file['gt_score']
        r, c = gt_score.shape
        targets = []
        for i in range(r):
            targets.append(gt_score[i][0])
        return targets
        
    def get_featureMatrix_and_targets(self, videoFileNamePath, videoMatFilePath):
        feature_matrix = self.getVideoFeatureMatrix(videoFileNamePath)
        targets = self.gt_mean_normalised_Targets(videoMatFilePath)
        
        return feature_matrix, targets


# To access this function call get_featureMatrix_and_targets fuction

#extract = Extract_Raw_RGB_features()
#fileName = '../SumMe/videos/Jumps.mp4'
#matFileName =  '../SumMe/GT/Jumps.mat'
#
#feature, targets = extract.get_featureMatrix_and_targets(fileName, matFileName)