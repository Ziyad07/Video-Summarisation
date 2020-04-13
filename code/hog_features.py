# -*- coding: utf-8 -*-
"""
Created on Thu May 31 23:09:02 2018

@author: ziyad
"""
import cv2
from skimage.feature import hog
from Extract_C3D import Extract_C3D
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm



file_names = glob.glob('../SumMe/videos/*.mp4')
extract = Extract_C3D()

for i in range(len(file_names)):
    
    video_as_numpy, frameCount = extract.vid2npy3(file_names[i])
    feature_matrix = np.zeros([3136])

    for index in tqdm(xrange(0, frameCount, 16)):
        img = Image.fromarray(video_as_numpy[index], 'RGB')
        img = np.asarray(img.convert('L'))
        hogHist, hogImage = hog(img, orientations=16, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
        reshape_hogHist = hogHist.reshape((1,-1))
        feature_matrix = np.vstack((feature_matrix, reshape_hogHist))
        
    new_fileName = file_names[i].split('/')[-1].split('.')[0]        
    np.save('../saved_numpy_arrays/hog_features/hog_' + new_fileName + '.npy', feature_matrix[1:])


#
#file_name = '../SumMe/videos/Valparaiso_Downhill.mp4'
#video_as_numpy, frameCount = extract.vid2npy3(file_name)
#img = Image.fromarray(video_as_numpy[1], 'RGB')
#img = np.asarray(img.convert('L'))
#hogHist, hogImage = hog(img, orientations=16, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
#reshape_hogHist = hogHist.reshape((1,-1))
#

c3d_feature = np.load('../saved_numpy_arrays/fv_Air_Force_One.npy')
hog_feature = np.load('../saved_numpy_arrays/hog_features/hog_Air_Force_One.npy')