from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import glob
import scipy.io as sio
from tensorflow.keras.layers import Dense, GlobalAveragePooling3D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

seed(1)
set_random_seed(2)

window_length = 16
batchsize=8
videos = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/*.npy')
videos.sort()

def binarizeTargets(gt_score):
    targets = []
    for i in range(gt_score.shape[0]):
        if gt_score[i][0] > 0.05:
            targets.append(1)
        else:
            targets.append(0)    
#    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))

    return targets

def getSampleTrainingData(videoArray, frame_targets):
    X = []
    Y = []
    for i in range(window_length, len(videoArray) - window_length):
            snippet2 = videoArray[i:i+window_length]
            target_for_next_frame2 = frame_targets[i+window_length]
            X.append(snippet2)
            Y.append(target_for_next_frame2)
    
    X = np.array(X)
    Y = np.array(Y)
    x = 0.15
    num_samples = int(x * len(X))
    idx = np.random.permutation(len(X))
    
    # try sorting before extracting snippets
#    idx.sort()
    X = X[idx]
    Y = Y[idx]
    
    new_X = X[:num_samples] / 255.0
    new_Y = Y[:num_samples]
    
    return new_X, new_Y

def getAllTrainingData():
    
    training_X = np.zeros([1, window_length, 112, 112, 3])
    training_Y = np.array([])
    for i in range(int(len(videos)*0.8)):
        current_video = np.load(videos[i])
        fileName = videos[i].split('/')[-1].split('.')[0]
        print(fileName)
    #    video_targets = np.load('../../saved_numpy_arrays/RGB_features/targets_' + fileName + '.npy')
        video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
        video_targets = video_mat_file['gt_score']
        
        binarizedTargets = binarizeTargets(video_targets)
#        print(len(binarizedTargets))
        video_targets = binarizedTargets#.squeeze(axis=-1)
        #COmment out line below when not using binary targets
        video_targets = np.expand_dims(video_targets, axis=-1)
        sampled_X, sampled_Y = getSampleTrainingData(current_video, video_targets)
        #take line below out 2hen using binary targets
        sampled_Y = np.squeeze(sampled_Y, axis=-1)
#        print(current_video.shape)
#        print(sampled_X.shape)
#        print(training_X.shape)
        training_X = np.concatenate((training_X, sampled_X), axis=0)
        training_Y = np.concatenate((training_Y, sampled_Y), axis=0)
        
        
    training_X = training_X[1:]
    
    return training_X, training_Y

def predictFeatures(model, feature_image):
    feature_model = Model(model.input, model.layers[-2].output)
    return feature_model.predict(feature_image)

def getTestingdata(fileName):
    current_video = np.load('../../saved_numpy_arrays/RGB_as_numpy/'+ fileName + '.npy')
    
    video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
    video_targets = video_mat_file['gt_score']
    binarizedTargets = binarizeTargets(video_targets)
    video_targets = binarizedTargets#.squeeze(axis=-1)

    
    X = []
    Y = []
    for i in range(window_length, len(current_video) - window_length):
            snippet2 = current_video[i:i+window_length]
            target_for_next_frame2 = video_targets[i+window_length]
            X.append(snippet2)
            Y.append(target_for_next_frame2)
    
    X = np.array(X) / 255.0
    Y = np.array(Y)    
    
    return X, Y

def getTestResults(videos, model, evals):
    for i in range(int(len(videos)*0.8), len(videos)):
        fileName_only = videos[i].split('/')[-1].split('.')[0]
        print(fileName_only)
        testing_X, testing_Y = getTestingdata(fileName_only)
#        testing_X = testing_X / 255.0
        
        print('got data ')
        
        predictions = model.predict(testing_X) * 20.0
        predictions1 = smooth(np.squeeze(predictions, axis=-1), 10)        
        predictions1 = np.expand_dims(predictions1, axis=-1)        
        
        np.save('../../saved_numpy_arrays/predictions/I3D/'+fileName_only+'.npy', predictions)
        
        evals.evaluateSumMe(predictions, fileName_only)
        evals.evaluateSumMe(predictions1, fileName_only)
        
        
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def save_model(model):
    model.save('I3D_model/Fine_tunned_model/i3d_rgb.h5')
    print('Model Saved')
    
def load_models(path):
    return load_model(path + '/i3d_rgb.h5')

def train_model():
    
    training_X, training_Y = getAllTrainingData()
    
    #load model here
    base_model = load_models('I3D_model/Original_model/112_dims')
    #ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3), classes=2)
    #base_model = Xception(weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_X, training_Y, batch_size=batchsize, epochs=16, validation_split=0.1, shuffle=True)
    
    # Save model
    save_model(model)
    
    return model


def load_fine_tuned_model():      
    model = load_models('I3D_model/Fine_tunned_model')
    print('Model Loaded')
    return model

#model = load_fine_tuned_model()
model = train_model()

from SumMeEvaluation import SumMeEvaluation
evals = SumMeEvaluation()
getTestResults(videos, model, evals)
