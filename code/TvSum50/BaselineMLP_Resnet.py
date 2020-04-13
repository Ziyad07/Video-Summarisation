import numpy as np
import glob
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
import scipy.io as sio
from kts.cpd_auto import cpd_auto
from select_key_shots import power_norm, create_key_shots, get_cps_number, getFPS_tvSum50
import pdb
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

saved_path = '../../saved_numpy_arrays/TvSum50/Resnet_features/'
files = glob.glob(saved_path+'*.npy')
test_files = glob.glob(saved_path+'testing/*.npy')
files.sort()
test_files.sort()
batchsize=8
window_size=2

def create_importance_scores(targets):
    targets = np.array(map(lambda x: np.around(x,1), targets))
    
    return targets

def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 1.9) else 0), gt_score))

    return targets

def create_cuboids(x, y, window_size):
    X = []
    Y = []
    for i in range(window_size, x.shape[0]-window_size-1):
        temp = x[i-window_size:i+window_size+1]
        X.append(temp)
        Y.append(y[i+1])
        
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y


def create_model(train_X, train_Y, model_type='binary'):
    
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
    model.fit(train_X, train_Y, batch_size=batchsize,
          epochs=8, validation_split=0.1, shuffle=True)
    
    return model


def sub_sample(x, y, fps):
    #Sample 2 fps
    final_x = np.zeros([1,x.shape[1]])
    final_y = []
    for i in range(0,x.shape[0]-fps,fps):
#        pdb.set_trace()
        first=np.expand_dims(x[i],axis=0)
        second=np.expand_dims(x[i+(fps/2)],axis=0)
        final_x = np.concatenate((final_x, first), axis=0)
        final_x = np.concatenate((final_x, second), axis=0)
        
        final_y = np.concatenate((final_y, y[i]), axis=0)
        final_y = np.concatenate((final_y, y[i+(fps/2)]), axis=0)
    
    return final_x[1:], final_y

def read_resnet_data(model_type, scaler):
    X_final = np.zeros([1, 2*window_size+1, 2048])
    Y_final = []
    for i in range(len(files)):
        fileName = files[i].split('/')[-1].split('.')[0]
        temp_x = np.load(saved_path+fileName+'.npy')
        
        numpy_file_targets = np.expand_dims(np.load('../../saved_numpy_arrays/TvSum50/ground_truth/training/'+fileName+'.npy'),axis=-1)
        temp_y = np.squeeze(numpy_file_targets, axis=-1)
        current_fps = getFPS_tvSum50(fileName) 
        
        temp_x, temp_y = sub_sample(temp_x, temp_y, current_fps)
        print(fileName)
        
#        normalised_features = np.array([power_norm(e) for e in temp_x])
        normalised_features = scaler.fit_transform(temp_x)
    
        if model_type =='categorical':
            temp_y = create_importance_scores(temp_y)
#            temp_y = np.squeeze(temp_y,axis=-1)
            
        else:
            temp_y = binarizeTargets(temp_y)
            
        X, Y = create_cuboids(normalised_features, temp_y, window_size)

#        
#        pdb.set_trace()
        X_final = np.concatenate((X_final, X), axis = 0)
        Y_final = np.concatenate((Y_final, Y), axis = 0)
        
        
    return X_final[1:], Y_final

def test_cps(model, model_type, scaler):
    all_cps=[]
    all_fileNames = []
    all_ks_targets = []
    for i in range(len(test_files)):
        current_file_x = np.load(test_files[i])
        fileName = test_files[i].split('/')[-1].split('.')[0]
        current_fps = getFPS_tvSum50(fileName) 
#        normalised_features = np.array([power_norm(e) for e in current_file_x])
        normalised_features  = scaler.transform(current_file_x)
        K = np.dot(normalised_features, normalised_features.T)
        
        m = get_cps_number(current_file_x.shape[0], current_fps) # avergae 5sec cps
        
        cps, scores = cpd_auto(K, m, 1)
#        cps = np.arange(0,current_file_x.shape[0], current_fps*5, dtype=int)
        X, _ = create_cuboids(normalised_features, np.zeros([normalised_features.shape[0]]), window_size)
        preds = model.predict(X)
#        pdb.set_trace()
        if model_type =='categorical':
            preds = np.argmax(preds, axis=1)
        print(fileName)
        targets, ks_targets = create_key_shots(preds, cps)
        
        all_fileNames.append(fileName)
        all_cps.append(cps)
        all_ks_targets.append(ks_targets)
        
        sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/Resnet/MLP/' + fileName+'.mat', dict(x=preds))
        
    sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/Resnet/MLP/AllData.mat', dict(fileName = all_fileNames, shot_boundaries=all_cps, targets=all_ks_targets))


#def test_baseline(model_type):
model_type = 'binary'
#    model_type = 'categorical'
scaler = MinMaxScaler()
X, Y = read_resnet_data(model_type, scaler)
if model_type =='categorical':
    Y=Y*10
    Y = to_categorical(Y)
    
#y_train = np.expand_dims(y_train, axis=1)
model = create_model(X,Y, model_type)

test_cps(model, model_type, scaler)

#test_baseline('binary')