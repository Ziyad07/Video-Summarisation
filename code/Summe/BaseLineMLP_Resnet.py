import numpy as np
import glob
import scipy.io as sio
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from SumMeEvaluation import SumMeEvaluation
from kts.cpd_auto import cpd_auto
from select_key_shots import power_norm, create_key_shots, get_cps_number
import pdb
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


saved_path = '../../saved_numpy_arrays/Resnet_features/'
files = glob.glob(saved_path+'*.npy')
mat_files = '../../SumMe/GT/'
files.sort()
batchsize=8
window_size=2


def callbacks():
    filepath = 'Save_best_training_models/ResNet/resnet_stream.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',
                                     save_weights_only=False)
    callbacks_list = [checkpoint]
    
    return callbacks_list

def create_importance_scores(targets):
    targets = np.array(map(lambda x: np.around(x,1), targets))
    
    return targets

def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))

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
    call_back = callbacks()
    model.fit(train_X, train_Y, batch_size=batchsize, callbacks=call_back,
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
    for i in range(int(len(files)*0.8)):
        fileName = files[i].split('/')[-1].split('.')[0]
        temp_mat_file = sio.loadmat(mat_files+fileName+'.mat')
        current_fps = int(temp_mat_file.get('FPS')[0][0])
        temp_x = np.load(saved_path+fileName+'.npy')
        temp_y = temp_mat_file.get('gt_score')
        
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

def test_cps(model, evals, model_type, scaler):
    f_score1=[]
    f_score2=[]
    for i in range(int(len(files)*0.8), len(files)):
        current_file_x = np.load(files[i])
        fileName = files[i].split('/')[-1].split('.')[0]
        current_fps = int(np.round(sio.loadmat(mat_files+fileName+'.mat').get('FPS')[0][0]))
        
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
#        pdb.set_trace()
#        print(np.count_nonzero(targets)/float(current_file_x.shape[0]))
#        print(np.count_nonzero(ks_targets)/float(current_file_x.shape[0]))
        f1_max, f1_mean = evals.evaluateSumMe(targets, fileName)       
        f2_max, f2_mean = evals.evaluateSumMe(ks_targets, fileName)       
        
        f_score1.append(f1_max)
        f_score2.append(f2_max)
        
    print "greedy mean: ", np.mean(f_score1)
    print "ks mean: ", np.mean(f_score2)

def test_baseline(model_type):
    #model_type = 'binary'
#    model_type = 'categorical'
    scaler = MinMaxScaler()
    X, Y = read_resnet_data(model_type, scaler)
    if model_type =='categorical':
        Y=Y*10
        Y = to_categorical(Y)
        
    #y_train = np.expand_dims(y_train, axis=1)
    _ = create_model(X,Y, model_type)
    
    model = load_model('Save_best_training_models/ResNet/resnet_stream.h5')
    evals = SumMeEvaluation()
    test_cps(model, evals, model_type, scaler)

test_baseline('binary')