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
videos = glob.glob('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/*.npy')
videos.sort()

df = DataFrame.from_csv("../../Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno_cols.tsv", sep="\t")
videoName_Category = {}
for i in xrange(0, df.shape[0], 20):
    videoName_Category[df.index[i]] = df['Category'][i]


def getTestingdata(fileName):
    video_as_numpy = np.load('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/testing/'+ fileName + '.npy')

    video_category = videoName_Category.get(fileName)    
    video_mat_file = np.load('../../saved_numpy_arrays/TvSum50/ground_truth/testing/'+ fileName+ ' '+video_category +'.npy')
    video_targets = video_mat_file
#    video_as_numpy = np.expand_dims(current_video, axis=-1)
#    video_as_numpy = current_video
    
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
    X = np.squeeze(X, axis=-1)
#    print X.shape
#    new_X, new_Y = pkm.samplePointsfromCluster(X, Y, 0.03)
    x = 0.01
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
    for i in range(len(videos)):
        current_video = np.load(videos[i])
        fileName = videos[i].split('/')[-1].split('.')[0]
        print fileName
    #    video_targets = np.load('../../saved_numpy_arrays/RGB_features/targets_' + fileName + '.npy')
        video_category = videoName_Category.get(fileName) 
        video_mat_file = np.load('../../saved_numpy_arrays/TvSum50/ground_truth/'+ fileName+' ' + video_category +'.npy')
        video_targets = video_mat_file
        
        video_as_numpy = np.expand_dims(current_video, axis=-1)
        sampled_X, sampled_Y = getSampleTrainingData(video_as_numpy, video_targets)
        #take line below out 2hen using binary targets
        sampled_Y = np.squeeze(sampled_Y, axis=-1)
#        sampled_X = np.squeeze(sampled_X, axis=-1) # New line added
#        print sampled_X.shape
#        print training_X.shape
        training_X = np.concatenate((training_X, sampled_X), axis=0)
        training_Y = np.concatenate((training_Y, sampled_Y), axis=0)
        
        
    training_X = training_X[1:]
    
    return training_X, training_Y
# Remember that these are uint8 in the numpy files
print 'here'



import glob
import scipy.io
def getTestResults(model):
    testing_files = glob.glob('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/testing/*.npy')
    
    for i in range(len(testing_files)):
        current_testing_file = testing_files[i]
        fileName = current_testing_file.split('/')[-1].split('.')[0]
        numpy_feature_matrix, Y = getTestingdata(fileName)
#        numpy_feature_matrix = np.squeeze(numpy_feature_matrix, axis=-1).shape # new line added
    #    numpy_feature_matrix = np.load(current_testing_file)  
        print fileName        
        preds = model.predict(numpy_feature_matrix)
        
        scipy.io.savemat('../../saved_numpy_arrays/TvSum50/predictions/fiveFrame_targets_RGB_features/' + fileName + '.mat', dict(x=preds))
        
        
        
def getTestResults3dAE(encoder_model, conv_3d_model):
    testing_files = glob.glob('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/testing/*.npy')
    
    for i in range(len(testing_files)):
        current_testing_file = testing_files[i]
        fileName = current_testing_file.split('/')[-1].split('.')[0]
        numpy_feature_matrix, Y = getTestingdata(fileName)

        print fileName  
        print numpy_feature_matrix.shape
        latent_vars = encoder_model.predict(numpy_feature_matrix)        
        preds = conv_3d_model.predict(latent_vars)
        
        scipy.io.savemat('../../saved_numpy_arrays/TvSum50/predictions/3d_ae/' + fileName + '.mat', dict(x=preds))
# 

#        
#training_X, training_Y = getAllTrainingData()

#np.save('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_X',training_X)
#np.save('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_Y',training_Y)
#training_X = np.load('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_X.npy')
#training_Y = np.load('../../saved_numpy_arrays/TvSum50/5frame_training_data/training_Y.npy')
#
#y_train = np.array(map(lambda q: (1 if (q >= 1.9) else 0), training_Y))
#y_test = np.array(map(lambda w: (1 if (w >= 2) else 0), y_test))


#
#trainingY_scaled = np.squeeze(trainingY_scaled, axis=-1)


from Computational_Graphs import  Computational_Graphs
from Save_Load_Model import Save_Load_Model
cg_CNN = Computational_Graphs()

#model, encoder = cg_CNN.Conv_3D_AE(training_X)
#model, encoder = cg_CNN.simple_Convolutional_AutoEncoder(training_X)
#model = cg_CNN.GRU_RNN(training_X, y_train)
#model = cg_CNN.eval_simpleCNN(training_X, y_train)
#
save_load_model = Save_Load_Model()
encoder_model = save_load_model.loadModelAndWeights('3D_AE/encoder_model')
conv_3d_model = save_load_model.loadModelAndWeights('3D_AE/3d_conv_model')
getTestResults3dAE(encoder_model, conv_3d_model)

#cg_CNN.plot_conv_weights(model, u'conv3d_1')
#save_load_model.saveModelAndWeights(model, '3D_AE/3d_conv_model')

#saveModelAndWeights(model) 

#model = loadModelAndWeights()
#getTestResults(model)

#trainX = np.load('3D_AE/latent_var/latent_vars_training.npy')
#trainY = np.load('3D_AE/latent_var/targets.npy')

#model = cg_CNN.eval_simpleCNN(trainX, trainY)   
    
    
