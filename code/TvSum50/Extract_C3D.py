# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:13:08 2018

@author: ziyad
"""

import cv2
import numpy as np
from C3D import C3D
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import imageio
import scipy.io as sio

# This script has been taylored to  suit the SumMe dataset for now

class Extract_C3D(object):

    def get_video_frame_count(self, fileName):
        cap = cv2.VideoCapture(fileName)
        videoFrameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        framesNotToConsider = videoFrameCount % 16
        frameCount = videoFrameCount    
        
        if framesNotToConsider > 0:
            frameCount = videoFrameCount - framesNotToConsider
            
        return frameCount
            
    def vid2npy3(self, fileName):
        cap = cv2.VideoCapture(fileName)
        videoFrameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        framesNotToConsider = videoFrameCount % 16
        frameCount = videoFrameCount    
        
        if framesNotToConsider > 0:
            frameCount = videoFrameCount - framesNotToConsider
            
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
    
        c3d = C3D()
        feature_vector = np.zeros([4096]) # size of returned vector from c3d
        for index in tqdm(xrange(0, frameCount, 16)):    
            video_segment = videoNumpyArray[index:index+16]
            inputSeq = np.expand_dims(video_segment, axis=0)
            video_segment_feature_vector = c3d.infer_layer(inputSeq)
#            feature_vector = np.append(feature_vector, video_segment_feature_vector)
            feature_vector = np.vstack((feature_vector, video_segment_feature_vector))
            
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
    
    def binarizeTargets(self, fileName):
        video_mat_file = self.readData(fileName)
        gt_score = video_mat_file['gt_score']
        r, c = gt_score.shape
        targets = []
        for i in range(r):
            if gt_score[i][0] > 0.0:
                targets.append(1)
            else:
                targets.append(0)
        return targets
        
    def getTargets(self, fileName, frameCount):
        binarizedTargets = self.binarizeTargets(fileName)
        singleTargetPer16Frames = []
        for i in xrange(0, frameCount, 16):
            framesToConsider = binarizedTargets[i:i+16]
            if sum(framesToConsider) >= 14: # if the sum of 16 frames are >=8 make target 1
                singleTargetPer16Frames.append(1)
            else:
                singleTargetPer16Frames.append(0)
                
        return singleTargetPer16Frames
    
    def get_featureMatrix_and_targets(self, videoFileNamePath, videoMatFilePath):
        feature_matrix, frameCount = self.getVideoFeatureMatrix(videoFileNamePath)
        targets = self.getTargets(videoMatFilePath, frameCount)
        
        return feature_matrix, targets


# To access this function call get_featureMatrix_and_targets fuction