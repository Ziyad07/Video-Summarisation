# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:00:53 2018

@author: ziyad
"""
import cv2
import numpy as np
import glob

def vid2npy3(fileName):
    cap = cv2.VideoCapture(fileName)
    videoFrameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    frameCount = videoFrameCount
    # frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    # frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frameHeight = 112
    frameWidth = 112
    buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
#    print buf.shape
    fc = 0
    ret = True
    print fileName
    while (fc < frameCount and ret):
        ret, frame = cap.read()
        try:    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray, (frameHeight, frameWidth))
        
            buf[fc] = resize
            fc += 1
        except:
            print 'Error'
            buf = buf[1:]
            continue

    cap.release()
    fileName_ex = fileName.split('/')[-1].split('.')[0]
#    print buf.shape
    np.save('../../saved_numpy_arrays/TvSum50/Videos_as_numpy/' + fileName_ex + '.npy', buf)
#    return buf, frameCount
    
files = glob.glob('../../Webscope_I4/ydata-tvsum50-v1_1/video/*.mp4')
for video in files:
    vid2npy3(video)

#vid2npy3('../../SumMe/videos/Cooking.mp4')
#temp = np.load('../../saved_numpy_arrays/Videos_as_numpy/Cooking.npy')
#
#import scipy.io as sio
#video_mat_file = sio.loadmat('../../SumMe/GT/Cooking.mat')