# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:21:36 2018

@author: ziyad
"""

from keras.models import model_from_json
from keras import Model
import h5py

# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)

model = model_from_json(open('model.json', 'r').read())
f = h5py.File('../models/sports1M_weights_tf.h5', mode='r')


#model = Model.load_weights(model, '../models/sports1M_weights_tf.h5')

model.load_weights('../models/sports1M_weights_tf.h5')
model.compile(loss='mean_squared_error', optimizer='sgd')


import cv2
import numpy as np

cap = cv2.VideoCapture('dM06AMFLsrc.mp4')

vid = []
while True:
    ret, img = cap.read()
    if not ret:
        break
    vid.append(cv2.resize(img, (171, 128)))
vid = np.array(vid, dtype=np.float32)


import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(vid[2000]/256)

X = vid[2000:2016, 8:120, 30:142, :].transpose((0, 1, 2, 3))
output = model.predict_on_batch(np.array([X]))
plt.plot(output[0][0])




print('Position of maximum probability: {}'.format(output[0].argmax()))
print('Maximum probability: {:.5f}'.format(max(output[0][0])))
print('Corresponding label: {}'.format(labels[output[0].argmax()]))

# sort top five predictions from softmax output
top_inds = output[0][0].argsort()[::-1][:5]  # reverse sort and take five largest items
print('\nTop 5 probabilities and labels:')
#_ =[print('{:.5f} {}'.format(output[0][0][i], labels[i])) for i in top_inds]





