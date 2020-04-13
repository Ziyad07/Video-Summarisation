from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import glob
import scipy.io as sio

seed(1)
set_random_seed(2)

window_length = 5
videos = glob.glob('../../saved_numpy_arrays/OpticalFlow_combined/*.npy')
videos.sort()


def getTestingdata(fileName):
    current_video = np.load('../../saved_numpy_arrays/OpticalFlow_combined/of_'+ fileName + '.npy')
#    x_flow = current_video[...,0]
#    y_flow = current_video[...,1]
#    combined_flow = x_flow + y_flow
    
    video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
    video_targets = video_mat_file['gt_score']
#    video_as_numpy = np.expand_dims(combined_flow, axis=-1)
    binarizedTargets = binarizeTargets(video_targets)
    video_targets = binarizedTargets#.squeeze(axis=-1)

    
    X = []
    Y = []
    for i in range(window_length, len(current_video) - window_length):
            snippet2 = current_video[i:i+window_length]
            target_for_next_frame2 = video_targets[i+window_length]
            X.append(snippet2)
            Y.append(target_for_next_frame2)
    
#    X = np.array(X) / 255.0
    X = np.array(X)# / 255.0
    Y = np.array(Y)    
    
    return X, Y

def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))

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
    x = 0.03
    num_samples = int(x * len(X))
    idx = np.random.permutation(len(X))
    idx.sort()
    X = X[idx]
    Y = Y[idx]
    
#    new_X = X[:num_samples] / 255.0
    new_X = X[:num_samples]# / 255.0
    new_Y = Y[:num_samples]
    
    return new_X, new_Y


def getAllTrainingData():
    
    training_X = np.zeros([1, 5, 112, 112, 2])
    training_Y = np.array([])
    for i in range(int(len(videos)*0.8)):
        current_video = np.load(videos[i])
        fileName = videos[i].split('/')[-1].split('.')[0].split('of_')[-1]
        print fileName
    #    video_targets = np.load('../../saved_numpy_arrays/RGB_features/targets_' + fileName + '.npy')
        video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
        video_targets = video_mat_file['gt_score']
        
        binarizedTargets = binarizeTargets(video_targets)
        video_targets = binarizedTargets#.squeeze(axis=-1)
        #COmment out line below when not using binary targets
        video_targets = np.expand_dims(video_targets, axis=-1)
#        x_flow = current_video[...,0]
#        y_flow = current_video[...,1]
#        combined_flow = x_flow + y_flow
        
#        video_as_numpy = np.expand_dims(combined_flow , axis=-1)
        
        sampled_X, sampled_Y = getSampleTrainingData(current_video, video_targets)
        #take line below out 2hen using binary targets
        sampled_Y = np.squeeze(sampled_Y, axis=-1)
        print video_targets.shape, ' ' , current_video.shape
#        print sampled_Y.shape
#        print training_Y.shape
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
import evaluate_metrics as emf 


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
    m.fit(training_X, training_Y, epochs=1, batch_size=32)
    
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
    json_file = open('5FrameTargets_opticalFlow/Binary/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model

    loaded_model.load_weights("5FrameTargets_opticalFlow/Binary/model.h5")
    print("Loaded model from disk")
    
    
    return loaded_model
    
def evalLoadedModel(loaded_model, X, Y):
        # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def getTestResults(videos, model, evals, mms):
    for i in range(int(len(videos)*0.8)+1, len(videos)):
        fileName_only = videos[i].split('/')[-1].split('.')[0].split('of_')[-1]
        print fileName_only    
        testing_X, testing_Y = getTestingdata(fileName_only)

        testing_X = np.reshape(testing_X, (testing_X.shape[0], dims[1] * dims[2] * dims[3] * dims[4]))
        testing_X_scaled = mms.transform(testing_X)
        testing_X = np.reshape(testing_X_scaled, (testing_X.shape[0], dims[1], dims[2], dims[3], dims[4]))
        
        print 'got data '
        predictions = model.predict(testing_X) * 20.0
        
#        evals.getSumMeEvaluationMetrics(predictions, fileName_only)
        evals.evaluateSumMe(predictions, fileName_only)


training_X, training_Y = getAllTrainingData()
dims = training_X.shape
training_X = np.reshape(training_X, (dims[0], dims[1] * dims[2] * dims[3] * dims[4]))
#
#training_Y = np.expand_dims(training_Y, axis=-1)
#
mms = MinMaxScaler().fit(training_X)
trainingX_scaled = mms.transform(training_X)

##testingY_scaled = mms_targets.transform(testing_Y)
#
##trainingY_scaled  = training_Y * 100
##testingY_scaled  = testing_Y * 100
#
#trainingY_scaled = np.squeeze(trainingY_scaled, axis=-1)
##testingY_scaled = np.squeeze(testingY_scaled, axis=-1)
#
#
##scores = model.evaluate(testing_X, testingY_scaled)
##print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
##predictions = model.predict(testing_X)
#
#model = eval_model(training_X, trainingY_scaled)
#    
#saveModelAndWeights(model)  

training_X = np.reshape(trainingX_scaled, (dims[0], dims[1], dims[2], dims[3], dims[4]))
from Computational_Graphs import  Computational_Graphs
from Save_Load_Model import Save_Load_Model
cg_CNN = Computational_Graphs()
model = cg_CNN.eval_simpleCNN(training_X, training_Y)
##model = loadModelAndWeights()
#
#
from SumMeEvaluation import SumMeEvaluation
evals = SumMeEvaluation()
#evals = emf.evaluate_metrics()
getTestResults(videos, model, evals, mms)
