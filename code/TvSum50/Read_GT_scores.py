# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:47:21 2018

@author: ziyad
"""

import scipy.io as sio
import numpy as np

filePath = 'ydata-tvsum50_test.mat'

matfile = sio.loadmat(filePath)
tvsum50_data = matfile['tvsum50'][0]


# Consider only the first video for now
saved_folder = '../../saved_numpy_arrays/TvSum50/ground_truth/'

for i in range(len(tvsum50_data)):
    video_name = str(tvsum50_data[i][0][0])
    category = str(tvsum50_data[i][1][0])
#    title = str(tvsum50_data[i][2][0])
    length = float(tvsum50_data[i][3][0])
    n_frames = int(tvsum50_data[i][4][0])
    user_anno = tvsum50_data[i][5][0]
    gt_score = tvsum50_data[i][6]
    print video_name
    np.save(saved_folder + video_name + ' ' + category, gt_score)

#tvsum50_data.sort(key=lambda x: x[0][0])


import glob
import shutil
from pandas import DataFrame

df = DataFrame.from_csv("../../Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno_cols.tsv", sep="\t")
videoName_Category = {}
for i in xrange(0, df.shape[0], 20):
    videoName_Category[df.index[i]] = df['Category'][i]
    
testing_files = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_features/testing/*.npy')   
def getTestingTargets():
    for i in range(len(testing_files)):
        fileName = testing_files[i].split('/')[-1].split('.')[0]
#        print fileName
        category = videoName_Category.get(fileName)
        new_file_name = fileName + ' ' + category + '.npy'
        print new_file_name
        shutil.move('../../saved_numpy_arrays/TvSum50/ground_truth/'+new_file_name, '../../saved_numpy_arrays/TvSum50/ground_truth/testing/'+new_file_name)

getTestingTargets()