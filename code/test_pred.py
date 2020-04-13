# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:36:18 2018

@author: ziyad
"""

import numpy as np

n_frames = 1000
window_length = 5
video = np.random.random((n_frames, 120, 160, 3))
frame_targets = np.random.random((n_frames,))

X = []
Y = []
for i in range(window_length, len(video) - window_length):
        snippet = video[i:i+window_length]
        target_for_next_frame = frame_targets[i+window_length]
        X.append(snippet)
        Y.append(target_for_next_frame)
X = np.array(X)
Y = np.array(Y)

print X.shape, Y.shape

#so that's all the snippets you can create from then video

#then, do decrease memory usage, just randomly sample x percent of them

x = 0.1
num_samples = int(x * len(X))
idx = np.random.permutation(len(X))
X = X[idx]
Y = Y[idx]

new_X = X[:num_samples]
new_Y = Y[:num_samples]

print new_X.shape, new_Y.shape