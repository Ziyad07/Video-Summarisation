import numpy as np
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, BatchNormalization, Bidirectional, Activation, Dropout, LSTM, Input, UpSampling1D
from keras.models import Sequential, Model
import glob
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from kts.cpd_auto import cpd_auto
from select_key_shots import create_key_shots, get_cps_number, create_cuboids, sub_sample, binarizeTargets, getFPS_tvSum50
import pdb

    
def read_features(files, path, window_size, scaler):    
    features = np.zeros([1,2*window_size+1, 1024])
    targets = []
    for i in range(len(files)):
        current_file_features = files[i]
        fileName = current_file_features.split('/')[-1].split('.')[0]
        numpy_file_features = np.load(current_file_features)
        numpy_file_targets = np.expand_dims(np.load(path+'targets/'+fileName+'.npy'),axis=-1)

        current_fps = getFPS_tvSum50(fileName)        
        
        numpy_file_targets = np.expand_dims(binarizeTargets(numpy_file_targets), axis=-1)
        temp_x, temp_y = sub_sample(numpy_file_features, numpy_file_targets, current_fps)
        normalised_features = scaler.fit_transform(temp_x)
        
        X, Y = create_cuboids(normalised_features, temp_y, window_size)
#        pdb.set_trace()
        features = np.concatenate((features, X), axis=0)
        targets = np.concatenate((targets, Y), axis=0)
        print(fileName)
        
    return features[1:], targets

def create_Conv_AE(training_X, batch_size=8, epochs=8):

    input_video = Input(shape=(training_X.shape[1], training_X.shape[2]))

    encoded = Dense(896, activation='relu')(input_video)
#    encoded = MaxPool1D((1, 2, 2))(encoded)
    encoded = Dense(768, activation='relu')(encoded)
    encoded = Conv1D(512, 3, padding='same')(encoded)
    
    latent = Dense(512)(encoded)
    
    decode_l1 = Dense(768, activation='relu')
#    decode_l2 = UpSampling1D((1, 2, 2))
    decode_l3 = Dense(896, activation='relu')
#    decode_l4 = Conv1D(1024, 3, padding='same')
    output_l = Dense(1024)
    
    decoded = decode_l1(latent)
#    decoded = decode_l2(decoded)
    decoded = decode_l3(decoded)
#    decoded = decode_l4(decoded)
    decoded = output_l(decoded)
    
    ae = Model(input_video, decoded)
    ae.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    encoder = Model(input_video, latent)
    
    ae.fit(training_X, training_X, batch_size=batch_size, epochs=epochs)

    return ae, encoder


def create_ConvRNN(trainingX, trainingY, num_epochs=8, num_batch=8):
        input_dims = trainingX.shape[2]
        
        model = Sequential()
#        model.add(Conv1D(256, 3, input_shape=(5, input_dims), padding='same'))
#        model.add(Conv1D(32, 3, padding='same'))
        
        model.add(Bidirectional(LSTM(32, 
                      input_shape=(5, input_dims),
                      return_sequences=True,
                      activation='tanh', 
                      recurrent_activation='hard_sigmoid', 
                      dropout=0.1,
                      recurrent_dropout=0.1)))
#        model.add(Conv1D(256, 3, input_shape=(5, input_dims), padding='same'))
        model.add(Conv1D(32, 3, padding='same'))
        
#        model.add(Dense(units=1024, kernel_initializer='he_normal', bias_initializer='random_normal'))
#        model.add(BatchNormalization())
#        model.add(Activation('relu'))
#        model.add(Dropout(0.4))
        
        model.add(Dense(units=256, kernel_initializer='he_normal', bias_initializer='random_normal'))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.4))
        
        model.add(Dense(units=256, kernel_initializer='he_normal', bias_initializer='random_normal'))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.4))
        
        model.add(Conv1D(32, 3, padding='same'))
        model.add(Bidirectional(LSTM(128, 
              return_sequences=True,
              activation='tanh', 
              recurrent_activation='hard_sigmoid', 
              dropout=0.1,
              recurrent_dropout=0.1)))
#        model.add(Dense(units=64, kernel_initializer='he_normal', bias_initializer='random_normal'))
#        model.add(BatchNormalization())
#        model.add(Activation('relu'))
#        model.add(Dropout(0.4))
        
        model.add(Flatten()) 
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trainingX, trainingY, 
                  epochs=num_epochs, 
                  batch_size=num_batch,
                  validation_split=0.1, shuffle=True)        
        
        return model
  
#def add_cps(cps, fps):
#    new_cps = []
#    new_cps.append(cps[0])
#    for i in range(len(cps)-1):
#        lower_bound = int(fps/2)
#        upper_bound = 10 * int(fps)
#        number_of_frames = cps[i+1]-cps[i]
#        if lower_bound < number_of_frames:#< upper_bound:        
#            new_cps.append(cps[i+1])
#        if number_of_frames > upper_bound:
#            added_point = cps[i] + int(number_of_frames / 2)
#            new_cps.append(added_point)
#    
#    new_cps = np.array(new_cps)
#    new_cps.sort()       
##    pdb.set_trace() 
#    return np.array(new_cps)

def test_cps(model_rgb, model_flow, enc_model_rgb, enc_model_flow, model_type, scaler_rgb, scaler_flow, files_rgb, files_flow, window_size):
    all_cps_flow=[]
    all_cps_rgb=[]
    all_cps_comb=[]
    all_fileNames = []
    all_ks_targets = []
    
    for i in range(len(files_rgb)):
        current_file_x_rgb = np.load(files_rgb[i])
        current_file_x_flow = np.load(files_flow[i])
        
        fileName = files_rgb[i].split('/')[-1].split('.')[0]
        current_fps = getFPS_tvSum50(fileName)
        print(fileName)
        
        normalised_features_flow = scaler_flow.transform(current_file_x_flow)
        normalised_features_rgb  = scaler_rgb.transform(current_file_x_rgb)
        normalised_features_comb = np.concatenate((normalised_features_flow, normalised_features_rgb))
        
        K = np.dot(normalised_features_flow, normalised_features_flow.T)
        m = get_cps_number(current_file_x_flow.shape[0], current_fps) # avergae 5sec cps
        cps, scores = cpd_auto(K, m, 1)
        
        K_r = np.dot(normalised_features_rgb, normalised_features_rgb.T)
        m_r = get_cps_number(current_file_x_rgb.shape[0], current_fps) # avergae 5sec cps
        cps_r, scores_r = cpd_auto(K_r, m_r, 1)
        
#        K_comb = np.dot(normalised_features_comb, normalised_features_comb.T)
#        m_comb = get_cps_number(current_file_x_rgb.shape[0], current_fps) # avergae 5sec cps
#        cps_comb, scores_comb = cpd_auto(K_comb, m_comb, 1)
#        cps = add_cps(cps, current_fps)
#        cps = np.arange(0,current_file_x.shape[0], current_fps*5, dtype=int)
        x_rgb, _ = create_cuboids(normalised_features_rgb, np.zeros([normalised_features_rgb.shape[0]]), window_size)
        x_flow, _ = create_cuboids(normalised_features_flow, np.zeros([normalised_features_flow.shape[0]]), window_size)
        
        
        if model_type=='AE':
            x_rgb = enc_model_rgb.predict(x_rgb)
            x_flow = enc_model_flow.predict(x_flow)
        
        preds_rgb = model_rgb.predict(x_rgb)
        preds_flow = model_flow.predict(x_flow)
        
        if preds_rgb.shape[0] > preds_flow.shape[0]:
            preds_rgb = preds_rgb[:preds_flow.shape[0]]
        
        preds = (preds_rgb+preds_flow) / 2.
        
        targets, ks_targets = create_key_shots(preds, cps)
        
#        pdb.set_trace()
        
        all_fileNames.append(fileName)
        all_cps_flow.append(cps)
        all_cps_rgb.append(cps_r)
#        all_cps_comb.append(cps_comb)
        
        all_ks_targets.append(ks_targets)
        
        
        
        sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/3D_AE/' + fileName + '_rgb.mat', dict(x=preds_rgb))
        sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/3D_AE/' + fileName + '_flow.mat', dict(x=preds_flow))
        sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/3D_AE/' + fileName + '_combined.mat', dict(x=preds))
        
    sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/3D_AE/AllData_flow_.mat', dict(fileName = all_fileNames, shot_boundaries=all_cps_flow, targets=all_ks_targets))
    sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/3D_AE/AllData_rgb_.mat', dict(fileName = all_fileNames, shot_boundaries=all_cps_rgb, targets=all_ks_targets))
#    sio.savemat('../../saved_numpy_arrays/TvSum50/predictions/I3D/3D_AE/AllData_comb_.mat', dict(fileName = all_fileNames, shot_boundaries=all_cps_comb, targets=all_ks_targets))
   
        

feature_rgb = 'RGB'
path_rgb = '../../saved_numpy_arrays/TvSum50/I3D_features/' + feature_rgb + '/'
files_rgb = glob.glob(path_rgb+'features/*.npy')
files_rgb.sort()

feature_flow = 'FLOW'
path_flow = '../../saved_numpy_arrays/TvSum50/I3D_features/' + feature_flow + '/'
files_flow = glob.glob(path_flow+'features/*.npy')
files_flow.sort()

window_size=2
batchsize=8        
model_type = 'AE' #AE

scaler_rgb = MinMaxScaler()
scaler_flow = MinMaxScaler()

x_rgb, y_rgb = read_features(files_rgb, path_rgb, window_size, scaler_rgb)
x_flow, y_flow = read_features(files_flow, path_flow, window_size, scaler_flow)

if model_type=='AE':
    model_rgb_ae, enc_rgb = create_Conv_AE(x_rgb)
    model_flow_ae, enc_flow = create_Conv_AE(x_flow)
    
    print("AE done training")
    new_x_rgb = enc_rgb.predict(x_rgb)
    new_x_flow = enc_flow.predict(x_flow)
    print("Latent Vars done")
    model_rgb = create_ConvRNN(new_x_rgb, y_rgb)
    model_flow = create_ConvRNN(new_x_flow, y_flow)
    
else:
    enc_rgb=0
    enc_flow=0
    model_rgb = create_ConvRNN(x_rgb, y_rgb)
    model_flow = create_ConvRNN(x_flow, y_flow)


files_flow = glob.glob(path_flow+'features/testing/*.npy')
files_flow.sort()
files_rgb = glob.glob(path_rgb+'features/testing/*.npy')
files_rgb.sort()

test_cps(model_rgb, model_flow, enc_rgb, enc_flow, model_type, scaler_rgb, scaler_flow, files_rgb, files_flow, window_size)
