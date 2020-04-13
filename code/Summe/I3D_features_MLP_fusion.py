import numpy as np
import glob
import scipy.io as sio
from BaseLineMLP_Resnet import sub_sample, create_cuboids
from select_key_shots import create_key_shots, get_cps_number
from sklearn.preprocessing import MinMaxScaler
import pdb
from SumMeEvaluation import SumMeEvaluation
from kts.cpd_auto import cpd_auto
from keras.layers import Dense, Activation, Flatten, merge, Input, Maximum, Average
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint

def callbacks(model_feature):
    filepath = 'Save_best_training_models/I3D_MLP/'+model_feature+'_stream.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=False)
    callbacks_list = [checkpoint]
    
    return callbacks_list

def create_model(feature, train_X, train_Y, model_type='binary'):
    
    model = Sequential()
    model.add(Dense(256, input_shape=(train_X.shape[1],train_X.shape[2],)))
    model.add(Activation('sigmoid'))
    model.add(Dense(256))
    model.add(Activation('sigmoid'))
    model.add(Flatten())
    if model_type =='binary':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    else:
        model.add(Dense(11, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        #mean_squared_error categorical_crossentropy
        
    call_back = callbacks(feature)
    model.fit(train_X, train_Y, batch_size=batchsize, callbacks=call_back,
          epochs=8, validation_split=0.1, shuffle=True)
    
    return model

def read_features(files, mat_files, path, window_size, scaler):    
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
        
    return features[1:], targets

def add_cps(cps, fps):
    new_cps = []
    new_cps.append(cps[0])
    for i in range(len(cps)-1):
        lower_bound = int(fps/2)
        upper_bound = 10 * int(fps)
        number_of_frames = cps[i+1]-cps[i]
        if lower_bound < number_of_frames:#< upper_bound:        
            new_cps.append(cps[i+1])
        if number_of_frames > upper_bound:
            added_point = cps[i] + int(number_of_frames / 2)
            new_cps.append(added_point)
    
    new_cps = np.array(new_cps)
    new_cps.sort()       
#    pdb.set_trace() 
    return np.array(new_cps)

def test_cps(model_rgb, model_flow, evals, model_type, scaler_rgb, scaler_flow, files_rgb, files_flow, mat_files, window_size):
    
    rgb = [] 
    flow = []
    comb_r = []
    comb_f = []
    
    for i in range(int(len(files_rgb)*0.8), len(files_rgb)):
        
        current_file_x_rgb = np.load(files_rgb[i])
        current_file_x_flow = np.load(files_flow[i])
        
        fileName = files_rgb[i].split('/')[-1].split('.')[0]
        current_fps = int(np.round(sio.loadmat(mat_files+fileName+'.mat').get('FPS')[0][0]))
        
        normalised_features_flow = scaler_flow.transform(current_file_x_flow)
        normalised_features_rgb  = scaler_rgb.transform(current_file_x_rgb)
        
        K = np.dot(normalised_features_flow, normalised_features_flow.T)
        m = get_cps_number(current_file_x_flow.shape[0], current_fps) # avergae 5sec cps
        cps, scores = cpd_auto(K, m, 1)
#        cps = add_cps(cps, current_fps)
        
        K_rgb = np.dot(normalised_features_rgb, normalised_features_rgb.T)
        m_rgb = get_cps_number(current_file_x_rgb.shape[0], current_fps) # avergae 5sec cps
        cps_rgb, scores_rgb = cpd_auto(K_rgb, m_rgb, 1)
#        cps_rgb = add_cps(cps_rgb, current_fps)
        
        x_rgb, _ = create_cuboids(normalised_features_rgb, np.zeros([normalised_features_rgb.shape[0]]), window_size)
        x_flow, _ = create_cuboids(normalised_features_flow, np.zeros([normalised_features_flow.shape[0]]), window_size)
        
        preds_rgb = model_rgb.predict(x_rgb)
        preds_flow = model_flow.predict(x_flow)
        
#        if model_type =='categorical':
#            preds = np.argmax(preds, axis=1)
        print(fileName)
        preds = (preds_rgb+preds_flow) / 2.
        
        targets, ks_targets_rgb = create_key_shots(preds_rgb, cps_rgb)
        targets, ks_targets_flow = create_key_shots(preds_flow, cps)
        
        targets, ks_targets_comb_r = create_key_shots(preds_flow, cps_rgb)
        targets, ks_targets_comb_f = create_key_shots(preds, cps)
        
#        np.save('../../saved_numpy_arrays/predictions/I3D/MLP/fusion/'+fileName+'_greedy_'+'.npy', targets)
        np.save('../../saved_numpy_arrays/predictions/I3D/MLP/fusion/'+fileName+'_ks_'+'.npy', ks_targets_comb_f)
#        pdb.set_trace()
        
        r_max, _ = evals.evaluateSumMe(ks_targets_rgb, fileName)     
        f_max, _ = evals.evaluateSumMe(ks_targets_flow, fileName)     
        comb_r_max, _ = evals.evaluateSumMe(ks_targets_comb_r, fileName)     
        comb_f_max, _ = evals.evaluateSumMe(ks_targets_comb_f, fileName)
        
        rgb.append(r_max)
        flow.append(f_max)
        comb_r.append(comb_r_max)
        comb_f.append(comb_f_max)
    
    print('rgb: ', np.mean(rgb))
    print('flow: ', np.mean(flow))
    print('comb rgb - cps: ', np.mean(comb_r))
    print('comb flow - cps: ', np.mean(comb_f))
        
def combine_flow_rgb(evals, shot_type):
    if shot_type == 'greedy':
        test_files_rgb = glob.glob('../../saved_numpy_arrays/predictions/I3D/MLP/fusion/*_greedy_.npy')
    else:
        test_files_rgb = glob.glob('../../saved_numpy_arrays/predictions/I3D/MLP/fusion/*_ks_.npy')
        
    test_files_rgb.sort()
    rgb = []
    for pred in range(len(test_files_rgb)):
        rgb_preds = np.load(test_files_rgb[pred])
        
        print(test_files_rgb[pred])
        
        if shot_type == 'greedy':    
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_rgb')[0].split('_greedy')[0]
        else:
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_rgb')[0].split('_ks')[0]
            
        r, r_mean = evals.evaluateSumMe(rgb_preds, fileName_only)

        rgb.append(r)
        
    print('fusion: ', np.mean(rgb))
        
#def test_fusion():
feature_rgb = 'RGB'
path_rgb = '../../saved_numpy_arrays/I3D_features/' + feature_rgb + '/'
files_rgb = glob.glob(path_rgb+'features/*.npy')
files_rgb.sort()

feature_flow = 'FLOW'
path_flow = '../../saved_numpy_arrays/I3D_features/' + feature_flow + '/'
files_flow = glob.glob(path_flow+'features/*.npy')
files_flow.sort()

mat_files = '../../SumMe/GT/'
window_size=2
batchsize=8        
model_type = 'binary'

scaler_rgb = MinMaxScaler()
scaler_flow = MinMaxScaler()

x_rgb, y_rgb = read_features(files_rgb, mat_files, path_rgb, window_size, scaler_rgb)
x_flow, y_flow = read_features(files_flow, mat_files, path_flow, window_size, scaler_flow)
_ = create_model('rgb', x_rgb, y_rgb, model_type)
_ = create_model('flow', x_flow, y_flow, model_type)

path = 'Save_best_training_models/I3D_MLP/'
model_rgb = load_model(path +'rgb_stream.h5')
model_flow = load_model(path +'flow_stream.h5')

evals = SumMeEvaluation()
test_cps(model_rgb, model_flow, evals, model_type, scaler_rgb, scaler_flow, files_rgb, files_flow, mat_files, window_size)

#shot_type = 'ks'
#combine_flow_rgb(evals, shot_type)
#print('\n')
#shot_type2 = 'greedy'
#combine_flow_rgb(evals, shot_type2)
#test_fusion()