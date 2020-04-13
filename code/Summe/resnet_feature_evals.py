import numpy as np
import glob
import scipy.io as sio
import evaluate_metrics as emf 
from sklearn.preprocessing import MinMaxScaler

from Computational_Graphs import  Computational_Graphs
from Save_Load_Model import Save_Load_Model
cg_CNN = Computational_Graphs()


summe_resnet_data = glob.glob('../../saved_numpy_arrays/Resnet_features/*.npy')
summe_resnet_data.sort()

def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))

    return targets

def getTrainingData():

    training_X = np.zeros([1, 2048])
    training_Y = np.array([])
    for i in range(int(len(summe_resnet_data)*0.8)):
        current_video = np.load(summe_resnet_data[i])
        fileName = summe_resnet_data[i].split('/')[-1].split('.')[0]
        print fileName
        video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')        
        video_targets = video_mat_file['gt_score']
        binarizedTargets = binarizeTargets(video_targets)
        
        training_X = np.concatenate((training_X, current_video), axis=0)
        training_Y = np.concatenate((training_Y, binarizedTargets), axis=0)
        
        
    training_X = training_X[1:]
    
    return training_X, training_Y

def getTestingdata(fileName):
    current_video = np.load('../../saved_numpy_arrays/Resnet_features/'+ fileName + '.npy')
    video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
    video_targets = video_mat_file['gt_score']
    
    return current_video, video_targets    
    

def getTestResults(videos, model, evals, mms):
    for i in range(int(len(videos)*0.8), len(videos)):
        fileName_only = videos[i].split('/')[-1].split('.')[0]
        print fileName_only    
        testing_X, testing_Y = getTestingdata(fileName_only)
        testing_X_scaled = mms.transform(testing_X)
        testing_X_scaled = np.reshape(testing_X_scaled, (testing_X_scaled.shape[0], time_steps, testing_X_scaled.shape[1]))
        
        
        print 'got data '
        predictions = model.predict(testing_X_scaled) * 20.0
        
        evals.getSumMeEvaluationMetrics(predictions, fileName_only)

time_steps = 1
trainingX, trainingY = getTrainingData()
mms = MinMaxScaler().fit(trainingX)
trainingX_scaled = mms.transform(trainingX)
trainingX_scaled = np.reshape(trainingX_scaled, (trainingX_scaled.shape[0], time_steps, trainingX_scaled.shape[1]))
#trainingX = np.reshape(trainingX, (trainingX.shape[0], time_steps, trainingX.shape[1]))
#trainingX_scaled = np.squeeze(trainingX_scaled, axis=1)


from sklearn.externals import joblib
scaler_filename = "Resnet_model/minMaxScaler.save"
joblib.dump(mms, scaler_filename) 


model = cg_CNN.simpleNeuralNetwork(trainingX_scaled, trainingY, num_epochs=20)
save_load = Save_Load_Model()
save_load.saveModelAndWeights(model, 'Resnet_model')

evals = emf.evaluate_metrics()
getTestResults(summe_resnet_data, model, evals, mms)


# And now to load...

#scaler = joblib.load(scaler_filename) 

