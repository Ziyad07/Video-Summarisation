import numpy as np
from Save_Load_Model import Save_Load_Model
import glob
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf

saveLoad = Save_Load_Model()
tvSumVids = glob.glob('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/224_dims/*.npy')
summeVids = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/224_dims/*.npy')

def predictFeatures(model, RGB_Frame):
    
    pred_model = Model(model.input, model.layers[-2].output)
    return pred_model.predict(RGB_Frame)


def readAllTVsumData():
    model = saveLoad.loadModelAndWeights('ResNetModel_fineTuned/Summe_TvSum')
    for i in tqdm(xrange(12, len(tvSumVids), 1)):
        current_vid = np.load(tvSumVids[i])        
        fileName = tvSumVids[i].split('/')[-1].split('.')[0]
        print i, ' ', fileName, ' ', current_vid.shape[0]
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        with tf.device('/gpu:0'):
            feature_matrix = predictFeatures(model, current_vid)
            np.save('../../saved_numpy_arrays/TvSum50/Resnet_features/'+fileName+'.npy', feature_matrix)

def readAllSummeData():
    model = saveLoad.loadModelAndWeights('ResNetModel_fineTuned/Summe_TvSum')
    for i in tqdm(range(len(summeVids))):
        current_vid = np.load(summeVids[i])        
        fileName = summeVids[i].split('/')[-1].split('.')[0]
        print i, ' ', fileName, ' ', current_vid.shape[0]
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        with tf.device('/gpu:0'):
            feature_matrix = predictFeatures(model, current_vid)
            np.save('../../saved_numpy_arrays/Resnet_features/'+fileName+'.npy', feature_matrix)
            
#readAllTVsumData()
readAllSummeData()