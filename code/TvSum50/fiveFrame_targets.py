from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import glob
import scipy.io as sio
from pandas import DataFrame
from perform_KMeans import perform_KMeans 

pkm = perform_KMeans()

seed(1)
set_random_seed(2)

window_length = 5
videos = glob.glob('../../saved_numpy_arrays/TvSum50/Videos_as_numpy/*.npy')
videos.sort()

df = DataFrame.from_csv("../../Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno_cols.tsv", sep="\t")
videoName_Category = {}
for i in xrange(0, df.shape[0], 20):
    videoName_Category[df.index[i]] = df['Category'][i]


def getTestingdata(fileName):
    current_video = np.load('../../saved_numpy_arrays/TvSum50/Videos_as_numpy/testing/'+ fileName + '.npy')

    video_category = videoName_Category.get(fileName)    
    video_mat_file = np.load('../../saved_numpy_arrays/TvSum50/ground_truth/testing/'+ fileName+ ' '+video_category +'.npy')
    video_targets = video_mat_file
    video_as_numpy = np.expand_dims(current_video, axis=-1)
    
    X = []
    Y = []
    for i in range(window_length, len(video_as_numpy) - window_length):
            snippet2 = video_as_numpy[i:i+window_length]
            target_for_next_frame2 = video_targets[i+window_length]
            X.append(snippet2)
            Y.append(target_for_next_frame2)
    
    X = np.array(X) / 255.0
    Y = np.array(Y)    
    
    return X, Y

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

    x = 0.2 #sample from complete array for clustering 
    num_samples = int(x * len(X))
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    
    new_X = X[:num_samples] / 255.0
    new_Y = Y[:num_samples]
    
    new_X = np.squeeze(new_X, axis=-1)
    print X.shape
    cluster_X, cluster_Y = pkm.samplePointsfromCluster(new_X, new_Y, 0.05) #sample points from cluster

#    return new_X, new_Y
    return cluster_X, cluster_Y


def getAllTrainingData():
    
    training_X = np.zeros([1, 5, 112, 112, 1])
    training_Y = np.array([])
    for i in range(len(videos)):
        current_video = np.load(videos[i])
        fileName = videos[i].split('/')[-1].split('.')[0]
        print i, ' ', fileName
    #    video_targets = np.load('../../saved_numpy_arrays/RGB_features/targets_' + fileName + '.npy')
        video_category = videoName_Category.get(fileName) 
        video_mat_file = np.load('../../saved_numpy_arrays/TvSum50/ground_truth/'+ fileName+' ' + video_category +'.npy')
        video_targets = video_mat_file
        
        video_as_numpy = np.expand_dims(current_video, axis=-1)
        sampled_X, sampled_Y = getSampleTrainingData(video_as_numpy, video_targets)
        #take line below out 2hen using binary targets
        sampled_Y = np.squeeze(sampled_Y, axis=-1)
        sampled_X = np.expand_dims(sampled_X, axis=-1)
#        print sampled_X.shape
#        print training_X.shape
        training_X = np.concatenate((training_X, sampled_X), axis=0)
        training_Y = np.concatenate((training_Y, sampled_Y), axis=0)
        
        
    training_X = training_X[1:]
    
    return training_X, training_Y
# Remember that these are uint8 in the numpy files
print 'here'


from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler


def comp_graph():
    m = Sequential()
    m.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=(5, 112, 112, 1)))
    m.add(MaxPooling3D())
    m.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
    m.add(MaxPooling3D())
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dense(1, activation='relu'))
    
    return m        

def eval_model(training_X, training_Y):
    m = comp_graph()
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.fit(training_X, training_Y, epochs=3, batch_size=32)
    
    return m

def saveModelAndWeights(model):
        # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def loadModelAndWeights():
    
    # load json and create model
    json_file = open('fiveFrameTargets/binary/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("fiveFrameTargets/binary/model.h5")
    print("Loaded model from disk")
    
    
    return loaded_model
    

import glob
import scipy.io
def getTestResults(model):
    testing_files = glob.glob('../../saved_numpy_arrays/TvSum50/Videos_as_numpy/testing/*.npy')
    
    for i in range(len(testing_files)):
        current_testing_file = testing_files[i]
        fileName = current_testing_file.split('/')[-1].split('.')[0]
        numpy_feature_matrix, Y = getTestingdata(fileName)
    #    numpy_feature_matrix = np.load(current_testing_file) 
        
        preds = model.predict(numpy_feature_matrix)
        print fileName
        scipy.io.savemat('../../saved_numpy_arrays/TvSum50/predictions/fiveFrame_targets/' + fileName + '.mat', dict(x=preds))
        
training_X, training_Y = getAllTrainingData()

#np.save('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_X',training_X)
#np.save('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_Y',training_Y)
#training_X = np.load('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_X.npy')
#training_Y = np.load('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_Y.npy')
#
#y_train = np.array(map(lambda q: (1 if (q >= 1.9) else 0), training_Y))
#y_test = np.array(map(lambda w: (1 if (w >= 2) else 0), y_test))




#training_Y = np.expand_dims(training_Y, axis=-1)
#
#mms_targets = MinMaxScaler().fit(training_Y)
#trainingY_scaled = mms_targets.transform(training_Y)
#testingY_scaled = mms_targets.transform(testing_Y)
#
##trainingY_scaled  = training_Y * 100
##testingY_scaled  = testing_Y * 100
#
#trainingY_scaled = np.squeeze(trainingY_scaled, axis=-1)


#model = eval_model(training_X, y_train)
    
#saveModelAndWeights(model)  

#model = loadModelAndWeights()
#getTestResults(model)




    
    
    
    