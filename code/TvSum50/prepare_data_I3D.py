import cv2
import numpy as np
import scipy.io as sio
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling3D, Dense, BatchNormalization, Dropout
import glob

WINDOW_LENGTH = 16
OVERLAP_PERCENTAGE = 0.5
batchsize = 8

train_videos = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/training/*.npy')
#videos.sort()

def get_cropped_videos(video):
    _, r, c, _ = video.shape
    V = 112
    X = []
    while len(X) < 5:
        x = np.random.randint(0, c)
        y = np.random.randint(0, r)
        if y + V <= r and x + V <= c:
            snippet = video[:, y:y + V, x:x + V, :]
            X.append(snippet)
    return np.array(X)


def rescale(vol):
    X_std = (vol - vol.min()) / (vol.max() - vol.min())
    return X_std * (1.0 - -1.0) + -1.0


def cropped_to_snippets(cropped_vols, p, binarizedTargets):
    X = []
    targets = []
    for cropped_vol in cropped_vols:
        snippets, Y = sample_snippets(cropped_vol, p, binarizedTargets)
        for (snippet, y) in zip(snippets, Y):
            X.append(snippet)
            targets.append(y)
    return np.array(X), np.array(targets)


def get_rgb_and_flow(video, flow, binarizedTargets, features='rgb'):
    p = 0.07
    FLOW = []   
    flow_targets = []
    RGB = []
    rgb_targets = []
    if features=='rgb':
#        video = np.array([resize(e) for e in video])
        video = rescale(video)
        cropped_rgb_videos = get_cropped_videos(video)
        RGB, rgb_targets = cropped_to_snippets(cropped_rgb_videos, p, binarizedTargets)

    if features=='flow':
#        flow = np.array([resize(e) for e in flow])
#        flow = np.clip(flow, -20, 20)  
        flow = rescale(flow)
        cropped_flow_videos = get_cropped_videos(flow)
        FLOW, flow_targets = cropped_to_snippets(cropped_flow_videos, p, binarizedTargets)    
        
    
    return RGB, rgb_targets, FLOW, flow_targets


def sample_snippets(video, p, binarizedTargets):
    step_size = int(np.ceil(WINDOW_LENGTH * (1 - OVERLAP_PERCENTAGE)))
    snippets = []
    Y = []
    for i in range(0, len(video) - WINDOW_LENGTH, step_size):
        snippet = video[i:i + WINDOW_LENGTH]
        target_for_next_frame2 = binarizedTargets[int(np.ceil(i+WINDOW_LENGTH)/2.)]
        
        snippets.append(snippet)
        Y.append(target_for_next_frame2)
        
    snippets = np.array(snippets)
    Y = np.array(Y)
    
    idx = np.random.permutation(len(snippets))
    #Sort
    idx.sort()
    num_to_sample = int(np.ceil(p * len(snippets)))
    return snippets[idx[:num_to_sample]], Y


def binarizeTargets(gt_score): # Needs to be fixed
    targets = np.array(map(lambda q: (1 if (q >= 1.9) else 0), gt_score))
    return targets


def save_model(model, feature):
    model.save('I3D_model/Fine_tunned_model/i3d_'+feature+'.h5')
    print('Model Saved')
    

def load_models(path, feature):
    model = load_model(path + '/i3d_'+feature+'.h5')
    print('model loaded')
    return model


def getAllTrainingData(features):
    if features == 'rgb':    
        training_X = np.zeros([1, WINDOW_LENGTH, 112, 112, 3])
    if features == 'flow':
        training_X = np.zeros([1, WINDOW_LENGTH, 112, 112, 2])
        
    training_Y = np.array([])    
        
    for i in range(len(train_videos)):
        fileName = train_videos[i].split('/')[-1].split('.')[0].split(' ')[0]
        print(i, len(train_videos), fileName)
        ground_truth = np.load('../../saved_numpy_arrays/TvSum50/ground_truth/training/'+fileName+'.npy')
        binarizedTargets = binarizeTargets(ground_truth)
        
        if features=='flow':
            flow = np.load('../../saved_numpy_arrays/TvSum50/Temp/FLOW/'+fileName+'.npy')
            rgb = np.array([])
            training_RGB, targets_RGB, training_FLOW, targets_FLOW = get_rgb_and_flow(rgb, flow, binarizedTargets, features)
            training_X = np.concatenate((training_X, training_FLOW), axis=0)
            training_Y = np.concatenate((training_Y, targets_FLOW), axis=0)
        
        if features=='rgb':
            rgb = np.load('../../saved_numpy_arrays/TvSum50/Temp/RGB/'+fileName+'.npy')
            flow = np.array([])

            training_RGB, targets_RGB, training_FLOW, targets_FLOW = get_rgb_and_flow(rgb, flow, binarizedTargets, features)
            training_X = np.concatenate((training_X, training_RGB), axis=0)
            training_Y = np.concatenate((training_Y, targets_RGB), axis=0)
        
    training_X = training_X[1:]
    
    return training_X, training_Y
def create_model(features):
        #load model here
    base_model = load_models('I3D_model/Original_model/112_dims', features)
    
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model
        
def train_model(features):
    
    training_X, training_Y = getAllTrainingData(features)
#    print(training_X.shape)
#    print(training_Y.shape)
    # checkpoint
    filepath="I3D_model/weights"+features+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model = create_model(features)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_X, training_Y, batch_size=batchsize,
              epochs=8, validation_split=0.1, shuffle=True, callbacks=callbacks_list)

    # Save model
    save_model(model, features)
    
    return model


def getTestingdata(fileName, features):
    if features == 'rgb':
        current_video = np.load('../../saved_numpy_arrays/TvSum50/Temp/RGB/'+ fileName + '.npy')

    if features == 'flow':
        current_video = np.load('../../saved_numpy_arrays/TvSum50/Temp/FLOW/'+fileName+'.npy')
    
    video_target_file = np.load('../../saved_numpy_arrays/TvSum50/ground_truth/training/'+ fileName +'.mat')
    binarizedTargets = binarizeTargets(video_target_file)
    video_targets = binarizedTargets#.squeeze(axis=-1)
    
    X = []
    Y = []
    for i in range(WINDOW_LENGTH, len(current_video) - WINDOW_LENGTH):
            snippet2 = current_video[i:i+WINDOW_LENGTH]
            target_for_next_frame2 = video_targets[i+WINDOW_LENGTH]
            X.append(snippet2)
            Y.append(target_for_next_frame2)
    
    X = np.array(X) 
    X = rescale(X)
    Y = np.array(Y)    
    
    return X, Y#, cps

def getTestResults(test_videos, model, features):
    for i in range(len(test_videos)):
        fileName_only = test_videos[i].split('/')[-1].split('.')[0]
        print(fileName_only)
        testing_X, testing_Y = getTestingdata(fileName_only, features)
#        testing_X = testing_X / 255.0
        print('got data ')
        targets = model.predict(testing_X) * 20.0
        np.save('../../saved_numpy_arrays/TvSum50/predictions/I3D/'+fileName_only+'_'+features+'.npy', targets)

def load_chkpnt_model(features):
    model = create_model(features)
    model.load_weights("I3D_model/weights.hdf5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

feature = 'rgb' #'rgb'
model = train_model(feature)
#model = load_models('I3D_model/Fine_tunned_model', feature)
#model = load_chkpnt_model( feature)
#train_videos = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/test/*.npy')
#getTestResults(test_videos, model, evals, feature)

