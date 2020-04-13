import numpy as np
from keras.layers import Activation, Dense, Flatten 
from keras.models import Sequential
import unicodedata
import scipy.io as sio
import pdb

tvsum50_data = sio.loadmat('ydata-tvsum50.mat')['tvsum50'][0]

def getFPS_tvSum50(fileName):
    fps = 29
    for i in range(len(tvsum50_data)):
        tvSum_filename = tvsum50_data[i][0][0]
        normalize_fileName = unicodedata.normalize('NFKD', tvSum_filename).encode('ascii','ignore')
        
        if fileName in normalize_fileName:
            fps = np.round(tvsum50_data[i][4][0][0] / tvsum50_data[i][3][0][0])
            
            return int(fps)

    return fps

def knapsack(weights, values, W):

    A = np.zeros([len(weights)+1,W+1])
    for j in range(len(weights)):
        for Y in range(W):
            if weights[j] > Y:
                A[j+1,Y+1] = A[j,Y+1]
            else:
                A[j+1,Y+1] = max( A[j,Y+1], values[j] + A[j,Y-weights[j]+1])
    
    best = A[-1,-1]
    
    amount = np.zeros([len(weights),1]);
    a = best;
    j = len(weights); 
    Y = W;
#    pdb.set_trace()
    while a > 0:
       while A[j,Y] == a:
           j = j - 1;
       amount[j] = a;
       Y = Y - weights[j]
       a = A[j,Y]
#       pdb.set_trace()
    return amount

def power_norm(v):
    normalised_vector = np.sign(v) * np.sqrt(np.abs(v))
    return normalised_vector

def create_key_shot_from_key_frame_eval(preds, cps):
    cps=np.insert(cps,0,0)
    
    scores=[]
    for i in range(len(cps)-1):
        interval_range=preds[cps[i]:cps[i+1]]
        interval_score = np.count_nonzero(interval_range) / float(len(interval_range))
        scores.append(interval_score)
        if np.count_nonzero(interval_range) > 0:
            preds[cps[i]:cps[i+1]] = 1
        
    idx = np.argsort(scores)
    
    final_targets = np.zeros(preds.shape)
    for ind, value in enumerate(idx):
#        final_targets[cps[idx[j-1]]:cps[idx[j]]] = j # check this line properly
        list_item_idx = cps[idx[ind]]
        list_item_cps_idx = np.where(cps==list_item_idx)[0][0]
        cps_point_1 = cps[list_item_cps_idx]
        cps_point_2 = cps[list_item_cps_idx-1]
        final_targets[cps_point_2:cps_point_1] = ind
        
#    pdb.set_trace()
    return final_targets
    

def create_key_shots(preds, cps):
    
#    preds = preds/20.0
    cps=np.insert(cps,0,0)
#    targets=np.zeros(preds.shape)
    averages = []
    snippet_weights=[]
    for i in range(len(cps)-1):
        averages.append(np.mean(preds[cps[i]:cps[i+1]]))
        snippet_weights.append(cps[i+1]-cps[i])
        
    # Call knapsack here
    W = int(len(preds)*0.15)
    amounts = knapsack(snippet_weights, averages, W)
    ks_targets = np.zeros(preds.shape)
#    previous_score=0
    for j in range(len(amounts)):
        if amounts[j] > 0:
            ks_targets [cps[j]:cps[j+1]] = amounts[j] #- previous_score
#            previous_score = amounts[j]
    
    idx = np.argsort(averages)
    
    final_targets = np.zeros(preds.shape)
    count = 0
    num_frames = preds.shape[0]
    for ind in range(len(idx)-1):
    #        final_targets[cps[idx[j-1]]:cps[idx[j]]] = j # check this line properly
        list_item_idx = cps[idx[ind]]
        list_item_cps_idx = np.where(cps==list_item_idx)[0][0]
        cps_point_1 = cps[list_item_cps_idx]
        cps_point_2 = cps[list_item_cps_idx+1]
        interval_len=cps_point_2-cps_point_1
        number_array = np.array(range(num_frames)[count:count+interval_len])
    
        if len(final_targets[cps_point_1:cps_point_2]) < len(number_array):
            limit = len(number_array) - len(final_targets[cps_point_1:cps_point_2])
            final_targets[cps_point_1:cps_point_2] = np.expand_dims(number_array[:-limit], axis=-1)
        elif len(final_targets[cps_point_1:cps_point_2]) > len(number_array):
            limit = len(final_targets[cps_point_1:cps_point_2]) - len(number_array)
            final_targets[cps_point_1:cps_point_2-limit] = np.expand_dims(number_array, axis=-1)
        else:
            final_targets[cps_point_1:cps_point_2] = np.expand_dims(number_array, axis=-1)
            
        count=count+interval_len

    return final_targets, ks_targets

def get_cps_number(number_of_frames, fps):
    lengh = int(number_of_frames / int(fps))
    number_cps = int(lengh/(5))
    
    return number_cps
    

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


def create_model(train_X, train_Y, model_type='binary', batchsize=8):
    
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