# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:07:24 2018

@author: ziyad
"""

import glob
import numpy as np
from ReadFileToNumpy2 import ReadFileToNumpy

rftn = ReadFileToNumpy()

fileNames = glob.glob('../../Webscope_I4/ydata-tvsum50-v1_1/video/*.mp4')

def RGB_numpy_arrays():
    for i in range(len(fileNames)):
        videoFile = fileNames[i]
#        videoMatFile = matFileNames[i]
        
        video, frameCount = rftn.vid2npy3(videoFile)
    #targets = extract.getTargets(videoMatFile, 4480)
        new_fileName = videoFile.split('/')[-1].split('.')[0]
        print(new_fileName)
    #save numpy arrays
        np.save('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/' + new_fileName + '.npy', video)
#        np.save('../saved_numpy_arrays/targets_' + new_fileName + '.npy', targets)

RGB_numpy_arrays()