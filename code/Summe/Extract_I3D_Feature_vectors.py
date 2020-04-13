import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Dense
import scipy.io as sio
import glob

WINDOW_LENGTH = 16
OVERLAP_PERCENTAGE = 0.5
batchsize = 8
#videos = ['../../saved_numpy_arrays/RGB_as_numpy/Jumps.npy']

def binarizeTargets(gt_score): 
    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))
    return targets

def prepare_data(video, binarizedTargets):
    step_size = int(np.ceil(WINDOW_LENGTH * (1 - OVERLAP_PERCENTAGE)))
    snippets = []
    Y = []
#    for i in range(0, len(video) - WINDOW_LENGTH, step_size):
    for i in range(0, len(video) - WINDOW_LENGTH, 1):
        snippet = video[i:i + WINDOW_LENGTH]
        target_for_next_frame2 = binarizedTargets[int(np.ceil(i+WINDOW_LENGTH)/2.)]
        
        snippets.append(snippet)
        Y.append(target_for_next_frame2)
        
    snippets = np.array(snippets)
    Y = np.array(Y)
    return snippets, Y
    
def run_through_data(videos, model, features_choice):
    for i in range(int(len(videos))):
        video = np.load(videos[i])
        if features_choice=='rgb':
            fileName = videos[i].split('/')[-1].split('.')[0]
        if features_choice=='flow':
            fileName = videos[i].split('/')[-1].split('.')[0].split('of_')[-1]
        print(fileName)
        video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
        video_targets = video_mat_file['gt_score']
        binarizedTargets = binarizeTargets(video_targets)
        snippets, targets = prepare_data(video, binarizedTargets)
        features = model.predict(snippets)
        if features_choice=='rgb':
            np.save('../../saved_numpy_arrays/I3D_features/RGB/features/'+fileName+'.npy', features)
            np.save('../../saved_numpy_arrays/I3D_features/RGB/targets/'+fileName+'.npy', targets)
        if features_choice=='flow':
            np.save('../../saved_numpy_arrays/I3D_features/FLOW/features/'+fileName+'.npy', features)
            np.save('../../saved_numpy_arrays/I3D_features/FLOW/targets/'+fileName+'.npy', targets)
        
        
        print(video.shape, snippets.shape, features.shape)
        
features_choice = 'flow'

if features_choice=='rgb':
    base_model = load_model('I3D_model/Fine_tunned_model/i3d_rgb.h5')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    videos = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/*.npy')
if features_choice=='flow':
    base_model = load_model('I3D_model/Fine_tunned_model/i3d_flow.h5')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    videos = glob.glob('../../saved_numpy_arrays/OpticalFlow_combined/*.npy')
    
run_through_data(videos, model, features_choice)