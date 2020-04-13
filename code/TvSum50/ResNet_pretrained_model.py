from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
import cv2
from pandas import DataFrame
from keras.callbacks import ModelCheckpoint
from Save_Load_Model import Save_Load_Model
saveLoad = Save_Load_Model()


import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 0.066) else 0), gt_score))

    return targets
   
#@threadsafe_generator
import keras
class My_Generator(keras.utils.Sequence):
    
    def __init__(self, fileNames, batch_size):
        self.fileNames = fileNames
        self.batch_size = batch_size

    def __len__(self):
        return 1000
#        return np.ceil(len(self.fileNames) / float(self.batch_size))    
    
    
    def __getitem__(self, idx):
        counter = np.random.randint(len(self.fileNames))
#        print 'counter: ', counter, ' '
        
        if counter == len(self.fileNames)+1:
            counter = 0
    
    #    print fileNames[counter]
        fileName = self.fileNames[counter].split('/')[-1].split('.')[0]
        batch_features = np.load('../../saved_numpy_arrays/RGB_as_numpy/224_dims/split/'+fileName+'.npy') / 255.0
        batch_labels = np.load('../../saved_numpy_arrays/RGB_as_numpy/224_dims/split/targets/'+fileName+'.npy')    
        batch_labels = binarizeTargets(batch_labels)
        
        while True:
#            print self.fileNames[counter]
            counter += 1
            for j in range(int((batch_features.shape[0]-self.batch_size)/self.batch_size)):
    #                print j
                x_batch = batch_features[j*self.batch_size:(j+1)*self.batch_size]
                y_batch = batch_labels[j*self.batch_size:(j+1)*self.batch_size] 
                
                return x_batch, y_batch
                    
@threadsafe_generator
def generator2(fileNames, batch_size):
    counter = 0
#        print 'counter: ', counter, ' '
    
    if counter == len(fileNames)+1:
        counter = 0

#    print fileNames[counter]
    fileName = fileNames[counter].split('/')[-1].split('.')[0]
    batch_features = np.load('../../saved_numpy_arrays/RGB_as_numpy/224_dims/split/'+fileName+'.npy') / 255.0
    batch_labels = np.load('../../saved_numpy_arrays/RGB_as_numpy/224_dims/split/targets/'+fileName+'.npy')    
    batch_labels = binarizeTargets(batch_labels)
    
    while True:
        print fileNames[counter]
        counter += 1
        for j in range(int((batch_features.shape[0]-batch_size)/batch_size)):
#                print j
            x_batch = batch_features[j*batch_size:(j+1)*batch_size]
            y_batch = batch_labels[j*batch_size:(j+1)*batch_size] 
            
            yield x_batch, y_batch


          
def sampleData(x_train, y_train, number_to_sample = 0.5):  
    
    num_samples = int(number_to_sample * len(x_train))
    idx = np.random.permutation(len(x_train))
    X = x_train[idx]
    Y = y_train[idx]    
    
    new_X = X[:num_samples] / 255.0
    new_Y = Y[:num_samples]

    return new_X, new_Y
    
def extractFeatures(model, sampleX):  
    new_model = Model(model.input, model.layers[-2].output)
    return new_model.predict(sampleX)

def model_checkpoint():
    # checkpoint
    filepath="ResNetModel_fineTuned/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    return callbacks_list

from keras.layers import BatchNormalization, Dropout, Activation
from keras import regularizers

import glob
import scipy.io as sio
#summe_video_data = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/224_dims/split/*.npy')
summe_video_data = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/224_dims/*.npy')
#summe_video_data.sort()
batchsize=16



trainFiles = glob.glob('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/224_dims/*.npy')
targets_folder_location = '../../saved_numpy_arrays/TvSum50/ground_truth/all_ground_truth/'
#model = saveLoad.loadModelAndWeights('ResNetModel_fineTuned/Summe')
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3), classes=2)
#base_model = Xception(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

#x = Dense(1024, 
#          activity_regularizer=regularizers.l1(0.01), 
#          kernel_regularizer=regularizers.l2(0.01))(x)
          
#x = Dense(2048)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = Dropout(0.4)(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#
##make all layers trainable
#for layer in base_model.layers:
#    layer.trainable=True
#    
#model.compile(optimizer='adam', loss='binary_crossentropy')
#model.fit_generator(generator=generator2(summe_video_data, batchsize), 
#                        validation_data=generator2(summe_video_data, batchsize), 
#                        validation_steps=10,
##                            callbacks=call_back,
#                        steps_per_epoch=500, 
#                        shuffle=True, epochs=2)
#for i, layer in  enumerate(base_model.layers):
#    print(i, layer.name)
    
#number_of_layers_non_trainable = len(model.layers) - 16
#for layer in model.layers[:number_of_layers_non_trainable]:
#    layer.trainable=False
#for layer in model.layers[number_of_layers_non_trainable:]:
#    layer.trainable=True

#trainFiles_targets = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/all_ground_truth/*.npy')
#trainFiles_targets = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/*.npy')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

def fineTune_TvSum50():
    
    training_X = np.zeros([1, 224, 224, 3])
    training_Y = np.array([])
    
    #training_X = []
    #training_X.append(smapled_X) for all i
    #training_X = np.array(training_X)
    for i in range(len(trainFiles)):
        fileName = trainFiles[i].split('/')[-1].split('.')[0]
        trainX = np.load(trainFiles[i])
        trainY = np.load(targets_folder_location + fileName + '.npy')
        y_train = np.array(map(lambda q: (1 if (q >= 1.9) else 0), trainY))
        print i, ' ', fileName
        sampled_X, sampled_Y = sampleData(trainX, y_train, number_to_sample=0.01)    
        training_X = np.concatenate((training_X, sampled_X), axis=0)
        training_Y = np.concatenate((training_Y, sampled_Y), axis=0)
        
    training_X = training_X[1:]
#    x_train, y_train, x_val, y_val = train_val_split(training_X, training_Y)
    model.fit(training_X, training_Y, batch_size=batchsize, epochs=10, validation_split=0.1, shuffle=True)
#    call_back = model_checkpoint()
#    model.fit_generator(generator=generator(x_train, y_train, batchsize), 
#                            validation_data=generator(x_val, y_val, batchsize), 
#                            validation_steps=x_val.shape[0]/batchsize,
#                            callbacks=call_back,
#                            steps_per_epoch=x_train.shape[0]/batchsize,
#                            shuffle=True, epochs=5)
    
def getTrainingData(summe_video_data):
    
    training_X = np.empty([1, 224, 224, 3])
    training_Y = np.array([])

    for i in range(len(summe_video_data)):
        fileName = summe_video_data[i].split('/')[-1].split('.')[0]
        trainX = np.load(summe_video_data[i])
        video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
        video_targets = video_mat_file['gt_score']
        
        y_train = np.array(map(lambda q: (1 if (q >= 0.05) else 0), video_targets))
        print i, ' ', fileName
        sampled_X, sampled_Y = sampleData(trainX, y_train, number_to_sample=0.02)  
        training_X = np.concatenate((training_X, sampled_X), axis=0)
        training_Y = np.concatenate((training_Y, sampled_Y), axis=0)
#        
    return training_X[1:], training_Y


from keras.callbacks import TensorBoard
from time import time

def fineTune_Summe():
    
    training_X, training_Y = getTrainingData(summe_video_data)
#    x_train, y_train, x_val, y_val = train_val_split(training_X, training_Y, val_split = 0.1)
    model.fit(training_X, training_Y, batch_size=batchsize, epochs=1, validation_split=0.1, shuffle=True)
#    call_back = model_checkpoint()
##    tensorboard = TensorBoard(log_dir="ResNetModel_fineTuned/logs/{}".format(time()))
#    train_gen = generator2(summe_video_data[5:], batchsize)
#    val_gen = generator2(summe_video_data[:4], batchsize)
#    model.fit_generator(generator=train_gen, 
#                            validation_data=val_gen, 
#                            validation_steps=80,
#                            callbacks=call_back,
#                            steps_per_epoch=1000, 
#                            shuffle=True, epochs=25,
#                            use_multiprocessing=False, workers=2)

    import gc
    gc.collect()
    return model

for i in range(1):    
    model = fineTune_Summe()
#fineTune_TvSum50()


def predictFeatures(model, feature_image):
    feature_model = Model(model.input, model.layers[-2].output)
    return feature_model.predict(feature_image)

#model = saveLoad.loadPretrainedModel('ResNetModel_fineTuned')
#faeture = predictFeatures(model, image_features)

#saveLoad.saveModelAndWeights(model, 'ResNetModel_fineTuned/Summe')
#model = saveLoad.loadPretrainedModel('ResNetModel_fineTuned')











#
#
#def split(fileNames):
#    for i in range(len(fileNames)):        
#        fileName = fileNames[i].split('/')[-1].split('.')[0]#.split('of_')[-1]
#        targets = sio.loadmat('../../SumMe/GT/'+fileName+'.mat')['gt_score']
##        batch_features = np.load('../../saved_numpy_arrays/RGB_as_numpy/224_dims/'+fileName+'.npy')
#        number_frames = targets.shape[0]
#        count = 0
#        for i in range(0, number_frames, 2000):
#            if (i+2000) > number_frames:
#                np.save('../../saved_numpy_arrays/RGB_as_numpy/224_dims/split/targets/'+fileName+'_'+str(count)+'.npy', targets[i:])
#            else:
#                np.save('../../saved_numpy_arrays/RGB_as_numpy/224_dims/split/targets/'+fileName+'_'+str(count)+'.npy', targets[i:i+2000])
#            count += 1
#
#        print fileName, ' ', targets.shape[0]
##        break
#        
#split(summe_video_data)










