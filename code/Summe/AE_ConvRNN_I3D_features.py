import numpy as np
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, BatchNormalization, Bidirectional, Activation, Dropout, LSTM, Input, UpSampling1D
from keras.models import Sequential, Model
from BaseLineMLP_Resnet import create_cuboids, sub_sample
import glob
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from SumMeEvaluation import SumMeEvaluation
from kts.cpd_auto import cpd_auto
from select_key_shots import create_key_shots, get_cps_number
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pdb

#feature = 'FLOW'
#features_used=feature.lower()
#path = '../../saved_numpy_arrays/I3D_features/' + feature + '/'
#mat_files = '../../SumMe/GT/'
#files = glob.glob(path+'features/*.npy')
#files.sort()
#window_size=2

def callbacks(model_feature):
    filepath = 'Save_best_training_models/AE_MLP/'+model_feature+'_stream.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=False)
    callbacks_list = [checkpoint]
    
    return callbacks_list

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
        
        X, Y = create_cuboids(normalised_features, temp_y, window_size)
#        pdb.set_trace()
        features = np.concatenate((features, X), axis=0)
        targets = np.concatenate((targets, Y), axis=0)
        print(fileName)
        
    return features[1:], targets

def create_Conv_AE(training_X, batch_size=8, epochs=8):

    input_video = Input(shape=(training_X.shape[1], training_X.shape[2]))
#    encoded = Conv1D(786, 3, padding='same')(input_video)
##    encoded = MaxPool1D((1, 2, 2))(encoded)
#    encoded = Conv1D(512, 3, padding='same')(encoded)
#    
#    latent = Dense(256, 3, padding='same')(encoded)
#    
#    decode_l1 = Conv1D(512, 3, padding='same')
##    decode_l2 = UpSampling1D((1, 2, 2))
#    decode_l3 = Conv1D(786, 3, padding='same')
#    output_l = Conv1D(1024, 3, padding='same')
#    encoded = Conv1D(1024, 3, padding='same')(input_video)
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


def create_ConvRNN(feature, trainingX, trainingY, num_epochs=8, num_batch=8):
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
        call_back = callbacks(feature)
        model.fit(trainingX, trainingY, 
                  epochs=num_epochs, 
                  batch_size=num_batch,
                  callbacks=call_back,
                  validation_split=0.1, shuffle=True)        
        
        return model
  
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

def test_cps(model_rgb, model_flow, enc_model_rgb, enc_model_flow, evals, model_type, scaler_rgb, scaler_flow, files_rgb, files_flow, mat_files, window_size):
    
    rgb = [] 
    flow = []
    comb_r = []
    comb_f = []
    
    for i in range(int(len(files_rgb)*0.8), len(files_rgb)):
        current_file_x_rgb = np.load(files_rgb[i])
        current_file_x_flow = np.load(files_flow[i])
        
        fileName = files_rgb[i].split('/')[-1].split('.')[0]
        current_fps = int(np.round(sio.loadmat(mat_files+fileName+'.mat').get('FPS')[0][0]))
        print(fileName)
        
        normalised_features_flow = scaler_flow.transform(current_file_x_flow)
        normalised_features_rgb  = scaler_rgb.transform(current_file_x_rgb)
        
        K = np.dot(normalised_features_flow, normalised_features_flow.T)
        m = get_cps_number(current_file_x_flow.shape[0], current_fps) # avergae 5sec cps
        cps, scores = cpd_auto(K, m, 1)
        cps = add_cps(cps, current_fps)
        
        K_rgb = np.dot(normalised_features_rgb, normalised_features_rgb.T)
        m_rgb = get_cps_number(current_file_x_rgb.shape[0], current_fps) # avergae 5sec cps
        cps_rgb, scores_rgb = cpd_auto(K_rgb, m_rgb, 1)
        cps_rgb = add_cps(cps_rgb, current_fps)
#        cps = np.arange(0,current_file_x.shape[0], current_fps*5, dtype=int)
        x_rgb, _ = create_cuboids(normalised_features_rgb, np.zeros([normalised_features_rgb.shape[0]]), window_size)
        x_flow, _ = create_cuboids(normalised_features_flow, np.zeros([normalised_features_flow.shape[0]]), window_size)
        
        
        if model_type=='AE':
            x_rgb = enc_model_rgb.predict(x_rgb)
            x_flow = enc_model_flow.predict(x_flow)
        
        preds_rgb = model_rgb.predict(x_rgb)
        preds_flow = model_flow.predict(x_flow)
        preds = (preds_rgb+preds_flow) / 2.
        
        targets, ks_targets_rgb = create_key_shots(preds_rgb, cps_rgb)
        targets, ks_targets_flow = create_key_shots(preds_flow, cps)
        
        targets, ks_targets_comb_r = create_key_shots(preds_flow, cps_rgb)
        targets, ks_targets_comb_f = create_key_shots(preds, cps)
#        np.save('../../saved_numpy_arrays/predictions/I3D/Conv_RNN/'+fileName+'_greedy_'+'.npy', targets)
        np.save('../../saved_numpy_arrays/predictions/I3D/Conv_RNN/'+fileName+'_ks_'+'.npy', ks_targets_comb_f)
#        pdb.set_trace()
#        evals.evaluateSumMe(targets, fileName)       
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
        test_files_rgb = glob.glob('../../saved_numpy_arrays/predictions/I3D/Conv_RNN/*_greedy_.npy')
#        test_files_flow= glob.glob('../../saved_numpy_arrays/predictions/I3D/Conv_RNN/*_greedy_flow.npy')
    else:
        test_files_rgb = glob.glob('../../saved_numpy_arrays/predictions/I3D/Conv_RNN/*_ks_.npy')
#        test_files_flow= glob.glob('../../saved_numpy_arrays/predictions/I3D/Conv_RNN/*_ks_flow.npy')
        
#    test_files_flow.sort()
    test_files_rgb.sort()
    rgb = []
#    flow = []
#    comb = []
    for pred in range(len(test_files_rgb)):
        rgb_preds = np.load(test_files_rgb[pred])
#        flow_preds = np.load(test_files_flow[pred])
        print(test_files_rgb[pred])
#        combined = (rgb_preds + flow_preds) / 2.0
        if shot_type == 'greedy':
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_greedy')[0]
        else:
            fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_ks')[0]
#        print('rgb: ')
        r, r_mean = evals.evaluateSumMe(rgb_preds, fileName_only)
#        print('flow: ')
#        f, f_mean = evals.evaluateSumMe(flow_preds, fileName_only)
#        print('combined: ')
#        c, c_mean = evals.evaluateSumMe(combined, fileName_only)

        rgb.append(r)
#        flow.append(f)
#        comb.append(c)
        
    print('rgb: ', np.mean(rgb))
#    print('flow: ', np.mean(flow))
#    print('comb: ', np.mean(comb))
        

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
model_type = 'AE' #AE

scaler_rgb = MinMaxScaler()
scaler_flow = MinMaxScaler()

x_rgb, y_rgb = read_features(files_rgb, mat_files, path_rgb, window_size, scaler_rgb)
x_flow, y_flow = read_features(files_flow, mat_files, path_flow, window_size, scaler_flow)
path = 'Save_best_training_models/AE_MLP/'

if model_type=='AE':
    model_rgb_ae, enc_rgb = create_Conv_AE(x_rgb)
    model_flow_ae, enc_flow = create_Conv_AE(x_flow)
    
    print("AE done training")
    new_x_rgb = enc_rgb.predict(x_rgb)
    new_x_flow = enc_flow.predict(x_flow)
    print("Latent Vars done")
    _ = create_ConvRNN('rgb',new_x_rgb, y_rgb)
    _ = create_ConvRNN('flow',new_x_flow, y_flow)

    model_rgb = load_model(path +'rgb_stream.h5')
    model_flow = load_model(path +'flow_stream.h5')
    
else:
    enc_rgb=0
    enc_flow=0
    _ = create_ConvRNN(x_rgb, y_rgb)
    _ = create_ConvRNN(x_flow, y_flow)
    model_rgb = load_model(path +'rgb_stream.h5')
    model_flow = load_model(path +'flow_stream.h5')


evals = SumMeEvaluation()
test_cps(model_rgb, model_flow, enc_rgb, enc_flow, evals, model_type, scaler_rgb, scaler_flow, files_rgb, files_flow, mat_files, window_size)
shot_type = 'ks'
#combine_flow_rgb(evals, shot_type)

#shot_type2 = 'greedy'
#combine_flow_rgb(evals, shot_type2)

#
#evals = SumMeEvaluation()
#scaler = MinMaxScaler()
#X, Y = read_features()
#model = create_ConvRNN(X, Y)
#test_cps(model, evals, scaler, features_used)
#shot_type = 'ks'
#combine_flow_rgb(shot_type)