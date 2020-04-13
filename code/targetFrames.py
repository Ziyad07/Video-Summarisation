# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:00:23 2018

@author: ziyad
"""

import scipy.io as sio
import cv2
import numpy as np

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
    


def readData():    
    air_force = sio.loadmat('Air_Force_One.mat')
    return air_force

def getTargets():
    air_force = readData()
    gt_score = air_force['gt_score']
    r, c = gt_score.shape
    targets = []
    for i in range(r):
        if gt_score[i][0] > 0.0:
            targets.append(1)
        else:
            targets.append(0)
        
def flattenVideo():
    air_force = readData()
    cap = cv2.VideoCapture("Air_Force_One.mp4")
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    
    segments = air_force['segments']
    amount_of_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    
    frames_per_second = air_force['FPS'][0][0]
    video_length = air_force['video_duration'][0][0]
    
    
    first_user_annotation = segments[:, 0][0]
    
    rows, cols = first_user_annotation.shape
    
    for row in range(rows):
        video_snippet = first_user_annotation[row,:]
        starting_point = int(video_snippet[0] * frames_per_second)
        ending_point = int(video_snippet[1] * frames_per_second)
#        print('here')
        for video_snippet_frames in xrange(starting_point, ending_point, 1):
            cap.set(1,video_snippet_frames); # Where frame_no is the frame you want
#            print(video_snippet_frames)
            ret, frame = cap.read() # Read the frame
            print(frame.shape)
#            out.write(frame)
#            cv2.imshow('window_name', frame) # show frame on window
#            if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
#               cap.release()
#               cv2.destroyAllWindows()
#               break
#            cv2.imshow('window_name',frame)
    
     #video_name is the video being called
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def splitTrainTest(video_array, targets):
    r,c = video_array.shape
    data_split = int(r * 0.7)
    training_data = video_array[:data_split,:]
    training_targets = targets[:data_split,:]
    testing_data = video_array[data_split:,:]
    testing_targets = targets[data_split:,:]    
    
    return training_data, training_targets, testing_data, testing_targets
    
    
def buildModel(training_data, training_targets, testing_data, testing_targets):
    timesteps = 75
    input_dim = (1080/2)*(1920/2) 
    hiddenLayer_dim = input_dim/4
        
    inputs = Input(shape=(timesteps, input_dim))
    hiddenLayer = LSTM(hiddenLayer_dim)(inputs)
    
    output = RepeatVector(timesteps)(hiddenLayer)
    output= LSTM(input_dim, return_sequences=True)(output)
    
    sequence_autoencoder = Model(inputs, output)
#    hidden_layer = Model(inputs, hiddenLayer)    

    sequence_autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')
    sequence_autoencoder.fit(training_data, training_targets,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(testing_targets, testing_targets))    
    

    
flattenVideo()
