import numpy as np
import glob
import scipy.io as sio
from select_key_shots import create_key_shots, get_cps_number, sub_sample, create_cuboids, create_model, getFPS_tvSum50
from sklearn.preprocessing import MinMaxScaler
from kts.cpd_auto import cpd_auto
import pdb

window_size=2
tvsum50_data = sio.loadmat('ydata-tvsum50_test_v1.mat')['tvsum50'][0]        
model_type = 'binary'

def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 1.9) else 0), gt_score))

    return targets

def read_features(files, path, scaler):    
    features = np.zeros([1,2*window_size+1, 1024])
    targets = []
    for i in range(len(files)):
        current_file_features = files[i]
        fileName = current_file_features.split('/')[-1].split('.')[0]
        numpy_file_features = np.load(current_file_features)
        numpy_file_targets = np.expand_dims(np.load(path+'targets/'+fileName+'.npy'),axis=-1)
        numpy_file_targets = np.squeeze(numpy_file_targets, axis=-1)
        current_fps = getFPS_tvSum50(fileName)         
        
        temp_x, temp_y = sub_sample(numpy_file_features, numpy_file_targets, current_fps)
        normalised_features = scaler.fit_transform(temp_x)
        X, Y = create_cuboids(normalised_features, temp_y, window_size)
        features = np.concatenate((features, X), axis=0)
        targets = np.concatenate((targets, Y), axis=0)
        print(fileName)
    
    targets = binarizeTargets(targets)
    return features[1:], targets

def test_cps(model, model_type, scaler, features_used, test_files):
    all_cps=[]
    all_fileNames = []
    all_ks_targets = []
    for i in range(len(test_files)):
        current_file_x = np.load(test_files[i])
        fileName = test_files[i].split('/')[-1].split('.')[0]
        
        current_fps = getFPS_tvSum50(fileName) 
        
        normalised_features  = scaler.transform(current_file_x)
        K = np.dot(normalised_features, normalised_features.T)
        m = get_cps_number(current_file_x.shape[0], current_fps) # avergae 5sec cp
        cps, scores = cpd_auto(K, m, 1)
        
#        cps = np.arange(0,current_file_x.shape[0], current_fps*5, dtype=int)
        X, _ = create_cuboids(normalised_features, np.zeros([normalised_features.shape[0]]), window_size)
        preds = model.predict(X)
#        np.save('../../saved_numpy_arrays/predictions/I3D/MLP/'+fileName+'_'+features_used+'.npy', preds)
        if model_type =='categorical':
            preds = np.argmax(preds, axis=1)
        print(fileName)
        
        targets, ks_targets = create_key_shots(preds, cps)

        all_fileNames.append(fileName)
        all_cps.append(cps)
        all_ks_targets.append(ks_targets)
        
        sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/' + fileName + '_' + features_used+'.mat', dict(x=preds))
        
    sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/AllData_'+features_used+'_.mat', dict(fileName = all_fileNames, shot_boundaries=all_cps, targets=all_ks_targets))
        
def combine_flow_rgb(shot_type):
    
    test_files_rgb = glob.glob('../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/*_rgb.mat')
    test_files_flow= glob.glob('../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/*_flow.mat')
        
    test_files_flow.sort()
    test_files_rgb.sort()
    for pred in range(len(test_files_rgb)):
        rgb_preds = sio.loadmat(test_files_rgb[pred]).get('x')
        flow_preds = sio.loadmat(test_files_flow[pred]).get('x')
        if rgb_preds.shape[0] > flow_preds.shape[0]:
            limit = rgb_preds.shape[0] - flow_preds.shape[0]
            rgb_preds = rgb_preds[:-limit]
        
        print(test_files_rgb[pred])
        
        if shot_type == 'greedy':    
            rgb_preds=rgb_preds/float(max(rgb_preds))
            flow_preds=flow_preds/float(max(flow_preds))   
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_rgb')[0].split('_greedy')[0]
        else:
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_rgb')[0]
            
        combined = (rgb_preds + flow_preds) / 2.0
        
        sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/'+fileName_only+'_combined'+'.mat', dict(x=combined))

def RunFunction(feature):
    features_used=feature.lower()
    path = '../../saved_numpy_arrays/TvSum50/I3D_features/' + feature + '/' # This needs to be fixed when we have the features
    files = glob.glob(path+'features/*.npy')
    files.sort()
    test_files = glob.glob(path+'features/testing/*.npy')
    
    scaler = MinMaxScaler()
    X,Y = read_features(files, path, scaler)
    model = create_model(X,Y, model_type)
    print('Getting test results for: ' + feature)
    test_cps(model, model_type, scaler, features_used, test_files)
    
RunFunction('FLOW')
RunFunction('RGB')

shot_type = 'ks'
combine_flow_rgb(shot_type)
