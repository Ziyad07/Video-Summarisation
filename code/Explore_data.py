# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:20:55 2018

@author: ziyad
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def readData(fileName):    
    video_mat_file = sio.loadmat(fileName)
    return video_mat_file

# Cooking
# St Maarten Landing
video_file_name = 'St Maarten Landing'
video_mat_file = readData('../SumMe/GT/' + video_file_name + '.mat')
    
gt_score = video_mat_file['gt_score']
segments = video_mat_file['segments'][0]
number_of_frames = int(video_mat_file['nFrames'][0][0])
frames_per_Sec = video_mat_file['FPS'][0][0]
user_score = video_mat_file['user_score']
gt_score = video_mat_file['gt_score']

plot_image = np.zeros((len(segments), number_of_frames))


for i in range(len(segments)):
    
    for j in range(len(segments[i])):        
        startFrame = int(segments[i][j][0]) * frames_per_Sec
        endFrame = int(segments[i][j][1]) * frames_per_Sec
        
        plot_image[i,startFrame:endFrame] = 1
    
    
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def plot_user_summaries():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(plot_image)
    ax.set_xlabel('Number of frames')
    ax.set_ylabel('Number of subjects')
    ax.set_aspect(2)
#    fig.savefig('equal.png')
    ax.set_aspect('auto')

def plot_user_summary_and_average_score():
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].imshow(plot_image)
    axarr[0].set_aspect(2)
    axarr[0].set_aspect('auto')
    axarr[0].set_ylabel('Number of users')
    axarr[0].set_title('User Summaries - ' + video_file_name)
    axarr[1].plot(gt_score)
    axarr[1].set_ylabel('Average Score')
    axarr[1].set_xlabel('Number of frames')    
    axarr[1].axhline(y=0.3, color='r', linestyle='-')    
    
#    threash_hold = np.full((1, number_of_frames), 0.4)
#    axarr[1].plot(threash_hold, color='r',  linestyle='-')
    
    f.savefig('User Summaries-' + video_file_name + '.png')

plot_user_summary_and_average_score()