# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:00:53 2018

@author: ziyad
"""
import cv2
import numpy as np
import glob

def perform_of(v):
    v = v.astype('uint8')
    f, r, c, d = v.shape
    previous_frame = cv2.cvtColor(v[0], cv2.COLOR_BGR2GRAY)
    flows = []
    for i in range(1, f):
        current_frame = cv2.cvtColor(v[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)    
        flows.append(flow)
        previous_frame = current_frame
        if (i % 1000) == 0:
            print(i)
        
    flows.append(flows[-1])
    return np.array(flows)#, np.array(combinedFlow)

def vid2npy_RGB(fileName):
    cap = cv2.VideoCapture(fileName)
    videoFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameCount = videoFrameCount
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    h = frameHeight if frameHeight <= frameWidth else frameWidth
    ratio = 128 / float(h)
    buf = []
    print(frameHeight, ' ', frameWidth)

#    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, frame = cap.read()
#            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret:
            im = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
            buf.append(im)
#            buf[fc] = frame
        else: 
            if frameCount != fc+1:
                ret = True
            print(fc)
            buf.append(buf[fc-1])
        fc += 1           
    cap.release()

    return np.array(buf), frameCount
    
    
def do_stuff():
    files = glob.glob('../../Webscope_I4/ydata-tvsum50-v1_1/video/*.mp4')
    for file_vid in files:

        fileName = file_vid.split('/')[-1].split('.')[0]
        print(fileName + '\n')
        video, fc = vid2npy_RGB(file_vid)
        #    video = np.array([resize(e) for e in video])
        np.save('../../saved_numpy_arrays/TvSum50/Temp/RGB/' + fileName + '.npy', video)
        flow = perform_of(video)
        flow = np.clip(flow, -20, 20)
        np.save('../../saved_numpy_arrays/TvSum50/Temp/FLOW/' + fileName + '.npy', flow)

do_stuff()

#files = glob.glob('../../SumMe/videos/*.mp4')
#from joblib import Parallel, delayed    
#Parallel(n_jobs=-1, backend="threading")(delayed(do_stuff)(fileName) for fileName in files)
