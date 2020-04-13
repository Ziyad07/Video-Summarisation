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

class C3D(object):

        def __init__(self):
            
            
def vid2npy3(fileName):
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


def vid2npy2(fileName):
    vid = imageio.get_reader(fileName, 'ffmpeg')
    x = []
    for frame in vid.iter_data():
        x.append(frame)
    x = np.array(x)
    return x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

def vid2npy(fileName):
        clip = VideoFileClip(fileName)
        n_frames = sum(1 for x in clip.iter_frames())
        s = None
        for frame in clip.iter_frames():
            s = frame.shape
            break        
        X = np.zeros((1, n_frames, s[0], s[1], s[2]))
        for i, frame in enumerate(clip.iter_frames()):
            X[0, i, :, :, :] = frame
                
        return X

def getFullVideoFeatureVector(videoNumpyArray, frameCount):

    c3d = C3D()
    feature_vector = []
    for index in tqdm(xrange(0, frameCount, 16)):    
        video_segment = videoNumpyArray[index:index+16]
        inputSeq = np.expand_dims(video_segment, axis=0)
        video_segment_feature_vector = c3d.infer_layer(inputSeq)
        feature_vector = np.append(feature_vector, video_segment_feature_vector)
        
    return feature_vector


fileName = 'dM06AMFLsrc.mp4'

#numpyArray = vid2npy(fileName)
#numpyArray2 = vid2npy2(fileName)
videoNumpyArray, frameCount = vid2npy3(fileName)

feature_vector = getFullVideoFeatureVector(videoNumpyArray, frameCount)
