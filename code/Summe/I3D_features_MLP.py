import numpy as np
import glob
import scipy.io as sio
from BaseLineMLP_Resnet import sub_sample, create_cuboids, create_model
from select_key_shots import create_key_shots, get_cps_number
from sklearn.preprocessing import MinMaxScaler
import pdb
from SumMeEvaluation import SumMeEvaluation
from kts.cpd_auto import cpd_auto

feature = 'RGB'
features_used=feature.lower()
path = '../../saved_numpy_arrays/I3D_features/' + feature + '/'
mat_files = '../../SumMe/GT/'
files = glob.glob(path+'features/*.npy')
files.sort()
window_size=2

def read_features():    
    features = np.zeros([1,2*window_size+1, 1024])
    targets = []
    for i in range(int(len(files)*0.8)):
        current_file_features = files[i]
        fileName = current_file_features.split('/')[-1].split('.')[0]
        numpy_file_features = np.load(current_file_features)
        numpy_file_targets = np.expand_dims(np.load(path+'targets/'+fileName+'.npy'),axis=-1)
        current_fps = int(sio.loadmat(mat_files+fileName+'.mat').get('FPS')[0][0])
        
        temp_x, temp_y = sub_sample(numpy_file_features, numpy_file_targets, current_fps)
        normalised_features = scaler.fit_transform(temp_x)
#        pdb.set_trace()
        X, Y = create_cuboids(normalised_features, temp_y, window_size)
#        pdb.set_trace()
        features = np.concatenate((features, X), axis=0)
        targets = np.concatenate((targets, Y), axis=0)
        print(fileName)
        
#    print(targets.shape)
#    print(features[1:].shape)
    return features[1:], targets

def test_cps(model, evals, model_type, scaler, features_used):
    for i in range(int(len(files)*0.8), len(files)):
        current_file_x = np.load(files[i])
        fileName = files[i].split('/')[-1].split('.')[0]
        current_fps = int(np.round(sio.loadmat(mat_files+fileName+'.mat').get('FPS')[0][0]))
        
        normalised_features  = scaler.transform(current_file_x)
        K = np.dot(normalised_features, normalised_features.T)
        
        m = get_cps_number(current_file_x.shape[0], current_fps) # avergae 5sec cps
        
        cps, scores = cpd_auto(K, m, 1)
#        cps = np.arange(0,current_file_x.shape[0], current_fps*5, dtype=int)
        X, _ = create_cuboids(normalised_features, np.zeros([normalised_features.shape[0]]), window_size)
        preds = model.predict(X)
#        np.save('../../saved_numpy_arrays/predictions/I3D/MLP/'+fileName+'_'+features_used+'.npy', preds)
        if model_type =='categorical':
            preds = np.argmax(preds, axis=1)
        print(fileName)
        
        targets, ks_targets = create_key_shots(preds, cps)
        np.save('../../saved_numpy_arrays/predictions/I3D/MLP/'+fileName+'_greedy_'+features_used+'.npy', targets)
        np.save('../../saved_numpy_arrays/predictions/I3D/MLP/'+fileName+'_ks_'+features_used+'.npy', ks_targets)
        
        evals.evaluateSumMe(targets, fileName)       
        evals.evaluateSumMe(ks_targets, fileName) 
        
def combine_flow_rgb(shot_type):
    if shot_type == 'greedy':
        test_files_rgb = glob.glob('../../saved_numpy_arrays/predictions/I3D/MLP/*_greedy_rgb.npy')
        test_files_flow= glob.glob('../../saved_numpy_arrays/predictions/I3D/MLP/*_greedy_flow.npy')
    else:
        test_files_rgb = glob.glob('../../saved_numpy_arrays/predictions/I3D/MLP/*_ks_rgb.npy')
        test_files_flow= glob.glob('../../saved_numpy_arrays/predictions/I3D/MLP/*_ks_flow.npy')
        
    test_files_flow.sort()
    test_files_rgb.sort()
    rgb = []
    flow = []
    comb = []
    for pred in range(len(test_files_rgb)):
        rgb_preds = np.load(test_files_rgb[pred])
        flow_preds = np.load(test_files_flow[pred])
        
        print(test_files_rgb[pred])
        
        if shot_type == 'greedy':    
            rgb_preds=rgb_preds/float(max(rgb_preds))
            flow_preds=flow_preds/float(max(flow_preds))   
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_rgb')[0].split('_greedy')[0]
        else:
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_rgb')[0].split('_ks')[0]
            
        combined = (rgb_preds + flow_preds) / 2.0
#        print('rgb: ')
        r, r_mean = evals.evaluateSumMe(rgb_preds, fileName_only)
#        print('flow: ')
        f, f_mean = evals.evaluateSumMe(flow_preds, fileName_only)
#        print('combined: ')
        c, c_mean = evals.evaluateSumMe(combined, fileName_only)

        rgb.append(r)
        flow.append(f)
        comb.append(c)
        
    print('rgb: ', np.mean(rgb))
    print('flow: ', np.mean(flow))
    print('comb: ', np.mean(comb))
        

model_type = 'binary'

scaler = MinMaxScaler()
X,Y = read_features()
model = create_model(X,Y, model_type)
    
evals = SumMeEvaluation()
test_cps(model, evals, model_type, scaler, features_used)
shot_type = 'greedy'
combine_flow_rgb(shot_type)
