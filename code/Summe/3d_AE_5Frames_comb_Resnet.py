from numpy.random import seed
from tensorflow import set_random_seed
import numpy as np
import glob
import scipy.io as sio
import evaluate_metrics as emf 
from Save_Load_Model import Save_Load_Model
save_load_model = Save_Load_Model()
from sklearn.externals import joblib

seed(1)
set_random_seed(2)

window_length = 5
videos = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/*.npy')
videos.sort()


def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))

    return targets
    
def getTestingdata_optical_flow(fileName):
    current_video = np.load('../../saved_numpy_arrays/OpticalFlow_combined/of_'+ fileName + '.npy')
    x_flow = current_video[...,0]
    y_flow = current_video[...,1]
    combined_flow = x_flow + y_flow
    
    video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
    video_targets = video_mat_file['gt_score']
    video_as_numpy = np.expand_dims(combined_flow, axis=-1)
    binarizedTargets = binarizeTargets(video_targets)
    video_targets = binarizedTargets#.squeeze(axis=-1)

    
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
    
def getTestingdata_5Frame(fileName):
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
    

def getTestingdata_resnet(fileName):
    current_video = np.load('../../saved_numpy_arrays/Resnet_features/'+ fileName + '.npy')
    video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
    video_targets = video_mat_file['gt_score']
    
    return current_video, video_targets    
    
    

def getTestResults_3d_ae(videos, encoder_model, conv_3d_model, resnet_model, optical_flow, evals, mms, time_steps=1):
    for i in range(int(len(videos)*0.8)+2, len(videos)):
        fileName_only = videos[i].split('/')[-1].split('.')[0]
        print fileName_only    
        
        testingX_of, testingY_of = getTestingdata_optical_flow(fileName_only)
        print 'got optical flow data'
        predictions_of = optical_flow.predict(testingX_of)        
        
        testing_X_5Frame, testing_Y = getTestingdata_5Frame(fileName_only)        
        print 'got data 5Frame'
        latent_vars = encoder_model.predict(testing_X_5Frame)
        print 'latent vars done'
        predictions_frame = conv_3d_model.predict(latent_vars) * 20.0
        
        testing_X, testing_Y = getTestingdata_resnet(fileName_only)
        testing_X_scaled = mms.transform(testing_X)
        testing_X_scaled = np.reshape(testing_X_scaled, (testing_X_scaled.shape[0], time_steps, testing_X_scaled.shape[1]))                
        print 'got data resnet'
        predictions_resnet = resnet_model.predict(testing_X_scaled) * 20.0
        
        predictions_frame = getSameSizeVectors(predictions_resnet, predictions_frame)
        predictions_of = getSameSizeVectors(predictions_resnet, predictions_of)
        
        predictions = getConcensusVote(predictions_frame, predictions_resnet, predictions_of)        
        predictions2 = getConcensusVote2(predictions_frame, predictions_resnet, predictions_of)
        evals.getSumMeEvaluationMetrics(predictions, fileName_only)
        evals.getSumMeEvaluationMetrics(predictions2, fileName_only)
        print '\n'

def getSameSizeVectors(resnet_array, frame_array):
    nFrames = resnet_array.shape[0]
    full_frame_targets = np.zeros((nFrames, 1))
    for i in range(len(frame_array)):
        full_frame_targets[i] = frame_array[i]
    return full_frame_targets

def getConcensusVote(preds_frame, preds_resnet, predictions_of):
    intersection = preds_frame * preds_resnet * predictions_of
    return intersection

def getConcensusVote2(preds_frame, preds_resnet, predictions_of):
    union= (2*preds_frame) + preds_resnet + predictions_of
    return union

encoder_model = save_load_model.loadModelAndWeights('3D_AE/encoder_model')
conv_3d_model = save_load_model.loadModelAndWeights('3D_AE/3d_conv_model')
resnet_model = save_load_model.loadModelAndWeights('Resnet_model')
optical_flow = save_load_model.loadModelAndWeights('5FrameTargets_opticalFlow/Binary')

evals = emf.evaluate_metrics()

scaler_filename = "Resnet_model/minMaxScaler.save"
mms = joblib.load(scaler_filename) 

getTestResults_3d_ae(videos, encoder_model, conv_3d_model, resnet_model, optical_flow, evals, mms)

#evals = emf.evaluate_metrics()
#getTestResults(summe_resnet_data, model, evals, mms)

