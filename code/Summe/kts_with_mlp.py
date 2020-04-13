import scipy.io as sio
from kts.cpd_auto import cpd_auto
import glob
from keras.layers import Dense, Input, Conv1D, BatchNormalization, Activation, Dropout, Flatten
from keras.models import Sequential
import numpy as np
from SumMeEvaluation import SumMeEvaluation
from select_key_shots import test_cps
evals = SumMeEvaluation()

# nuber of change points
m = 20
batchsize=8
features = glob.glob('../../saved_numpy_arrays/I3D_features/FLOW/features/*.npy')
features.sort()
targets_path = '../../SumMe/GT/'


def power_norm(v):
    normalised_vector = np.sign(v) * np.sqrt(np.abs(v))
    return normalised_vector

def binarizeTargets(gt_score):    
    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))

    return targets

def mlp(train_X, train_Y):
    input_dims = train_X.shape[1]
    model = Sequential()
#    model.add(Conv1D(512, 3, input_shape=(1, input_dims), padding='same'))
    model.add(Dense(256, input_shape=(input_dims,)))
    
#    model.add(Flatten())
    
    model.add(Dense(units=256, kernel_initializer='he_normal', bias_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.4))    
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_X, train_Y, batch_size=batchsize,
              epochs=16, validation_split=0.1, shuffle=True)

    return model

train_X = np.zeros([1, 1024])
train_Y = []
for i in range(int(len(features)*0.8)):
    fileName = features[i].split('/')[-1].split('.')[0]
    current_file_x = np.load(features[i])
    current_file_y = sio.loadmat(targets_path+fileName+'.mat').get('gt_score')
    print(fileName)
    new_targets=binarizeTargets(current_file_y)
#    K = np.dot(current_file_x, current_file_x.T)
#    cps, scores = cpd_auto(K, 2*m, 1)
#    new_targets = np.zeros(current_file_y.shape)
#    for i in range(len(cps)-1):
#        if np.count_nonzero(current_file_y[cps[i]:cps[i+1]]) > 0:
#            new_targets[cps[i]:cps[i+1]] = 1
#    break
    normalised_features = np.array([power_norm(e) for e in current_file_x])
#    new_targets = np.squeeze(new_targets, axis=-1)
    
    train_X = np.concatenate((train_X, normalised_features), axis=0)
    train_Y = np.concatenate((train_Y, new_targets), axis=0)
    
train_Y = train_Y[:train_X.shape[0]]
#train_X = np.expand_dims(train_X, axis=1)

model = mlp(train_X, train_Y)
test_cps(model)

def test():
    for i in range(int(len(features)*0.8), len(features)):
        current_file_x = np.load(features[i])
        fileName = features[i].split('/')[-1].split('.')[0]
        normalised_features = np.array([power_norm(e) for e in current_file_x])
        preds = model.predict(normalised_features)
        K = np.dot(current_file_x, current_file_x.T)
        cps, scores = cpd_auto(K, 2*m, 1)
        
        for i in range(len(cps)-1):
            if any(c > 0.3 for c in preds[cps[i]:cps[i+1]]):
                preds[cps[i]:cps[i+1]] = max(preds[cps[i]:cps[i+1]])
        
        print(fileName)
        evals.evaluateSumMe(preds, fileName)
        
    return preds

#preds = test()
