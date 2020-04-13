from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import glob
import scipy.io as sio

seed(1)
set_random_seed(2)

window_length = 5
videos = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/*.npy')
videos.sort()


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
    x = 0.05
    num_samples = int(x * len(X))
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    
    new_X = X[:num_samples] / 255.0
    new_Y = Y[:num_samples]
    
    return new_X, new_Y


def getAllTrainingData():
    
    training_X = np.zeros([1, 5, 112, 112, 3])
    training_Y = np.array([])
    for i in range(int(len(videos)*0.8)):
        current_video = np.load(videos[i])
        fileName = videos[i].split('/')[-1].split('.')[0]
        print fileName
    #    video_targets = np.load('../../saved_numpy_arrays/RGB_features/targets_' + fileName + '.npy')
        video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
        video_targets = video_mat_file['gt_score']
        
        binarizedTargets = binarizeTargets(video_targets)
        video_targets = binarizedTargets#.squeeze(axis=-1)
        #COmment out line below when not using binary targets
        video_targets = np.expand_dims(video_targets, axis=-1)
        
        sampled_X, sampled_Y = getSampleTrainingData(current_video, video_targets)
        #take line below out 2hen using binary targets
        sampled_Y = np.squeeze(sampled_Y, axis=-1)
#        print current_video.shape
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
import evaluate_metrics as emf 


def comp_graph():
    m = Sequential()
    m.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=(5, 112, 112, 3)))
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
    
def evalLoadedModel(loaded_model, X, Y):
        # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def getTestResults(videos, model, evals, graph='None'):
    for i in range(int(len(videos)*0.8), len(videos)):
        fileName_only = videos[i].split('/')[-1].split('.')[0]
        print fileName_only    
        testing_X, testing_Y = getTestingdata(fileName_only)
#        testing_X = testing_X / 255.0

        if graph == 'GRU':
            num_samples = testing_X.shape[0]
            testing_X = testing_X.reshape(num_samples, 5, 37632)
        
        print 'got data '
        predictions = model.predict(testing_X) * 20.0
        
        evals.getSumMeEvaluationMetrics(predictions, fileName_only)


def getTestResults_3d_ae(videos, encoder_model, conv_3d_model, evals):
    for i in range(int(len(videos)*0.8)+3, len(videos)):
        fileName_only = videos[i].split('/')[-1].split('.')[0]
        print fileName_only    
        testing_X, testing_Y = getTestingdata(fileName_only)
#        testing_X = testing_X / 255.0
        
        print 'got data '
        latent_vars = encoder_model.predict(testing_X)
        print 'latent vars done'
        predictions = conv_3d_model.predict(latent_vars) * 20.0
        print predictions.shape
        predictions1 = smooth(np.squeeze(predictions, axis=-1), 10)        
        print predictions1.shape
        predictions1 = np.expand_dims(predictions1, axis=-1)        
        
        np.save('../../saved_numpy_arrays/predictions/3D_AE/'+fileName_only+'.npy', predictions)
        
        evals.evaluateSumMe(predictions, fileName_only)
        evals.evaluateSumMe(predictions1, fileName_only)
#        evals.getSumMeEvaluationMetrics(predictions, fileName_only)
def smooth(y, box_pts):
#    box = np.array([0,1,1,1,2,5,2,1,1,1,0])/15.0
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



from Computational_Graphs import  Computational_Graphs
from Save_Load_Model import Save_Load_Model
cg_CNN = Computational_Graphs()

#training_X, training_Y = getAllTrainingData()
#model = eval_model(training_X, training_Y)  
#model = cg_CNN.GRU_RNN(training_X, training_Y)
#model = cg_CNN.eval_simpleCNN(training_X, training_Y)
#model, encoder_model = cg_CNN.Conv_3D_AE(training_X)


save_load_model = Save_Load_Model()
#save_load_model.saveModelAndWeights(encoder_model, '3D_AE/encoder_model')

#latent_vars = encoder_model.predict(training_X)
#np.save('3D_AE/latent_vars/latent_vars_training.npy', latent_vars)
#np.save('3D_AE/latent_vars/targets.npy', training_Y)
#model = save_load_model.loadModelAndWeights('3D_AE/encoder_model')
#predictions = encoder.predict(training_X)
#training_X = np.load('3D_AE/latent_vars/latent_vars_training.npy')
#targets = np.load('3D_AE/latent_vars/targets.npy')
#conv_3d_model = cg_CNN.eval_simpleCNN(latent_vars, training_Y)
#save_load_model.saveModelAndWeights(conv_3d_model, '3D_AE/3d_conv_model')

#
encoder_model = save_load_model.loadModelAndWeights('3D_AE/encoder_model')
conv_3d_model = save_load_model.loadModelAndWeights('3D_AE/3d_conv_model')


from SumMeEvaluation import SumMeEvaluation
evals = SumMeEvaluation()
#evals = emf.evaluate_metrics()
getTestResults_3d_ae(videos, encoder_model, conv_3d_model, evals)
#getTestResults(videos, model, evals)












