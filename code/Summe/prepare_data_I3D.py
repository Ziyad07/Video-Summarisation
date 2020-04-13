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

videos = glob.glob('../../SumMe/videos/*.mp4')
videos.sort()

def vid2npy_RGB(fileName):
    cap = cv2.VideoCapture(fileName)
    videoFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#    print('actual Frame Count: ', videoFrameCount)
    frameCount = videoFrameCount
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
#        print 'here'
    while (fc < frameCount and ret):
        ret, frame = cap.read()
#            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret:
            buf[fc] = frame
        else: 
            if frameCount != fc+1:
                ret = True
            print(fc)
            buf[fc] = buf[fc-1]
#                fc += 1
        fc += 1           
    cap.release()
#    print('read Frame count: ', fc)
    return buf, frameCount

def perform_of(v):
    v = v.astype('uint8')
    f, r, c, d = v.shape
    previous_frame = cv2.cvtColor(v[0], cv2.COLOR_BGR2GRAY)
    flows = []
#    combinedFlow = []
    for i in range(1, f):
        current_frame = cv2.cvtColor(v[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#        xflow = flow[..., 0]
#        yflow = flow[..., 1]
        flows.append(flow)
        previous_frame = current_frame
    print(' optical flow done')
    return np.array(flows)#, np.array(combinedFlow)

def resize(im):
    r, c, _ = im.shape
    h = r if r <= c else c
    ratio = 128 / float(h)
    return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)


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
    p = 0.25
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


def binarizeTargets(gt_score): 
    targets = np.array(map(lambda q: (1 if (q >= 0.05) else 0), gt_score))
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
        
    for i in range(int(len(videos)*0.8)):
        fileName = videos[i].split('/')[-1].split('.')[0]
        print(fileName)
        video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
        video_targets = video_mat_file['gt_score']
        binarizedTargets = binarizeTargets(video_targets)
        
        if features=='flow':
#            vid, frameCount = vid2npy_RGB(videos[i])
#            flow = perform_of(vid)
            flow = np.load('../../saved_numpy_arrays/Temp/FLOW/'+fileName+'.npy')
            rgb = np.array([])
            training_RGB, targets_RGB, training_FLOW, targets_FLOW = get_rgb_and_flow(rgb, flow, binarizedTargets, features)
#            print(training_X.shape)
#            print(training_FLOW.shape)
        
            training_X = np.concatenate((training_X, training_FLOW), axis=0)
            training_Y = np.concatenate((training_Y, targets_FLOW), axis=0)
        
        if features=='rgb':
#            rgb, frameCount = vid2npy_RGB(videos[i])
            rgb = np.load('../../saved_numpy_arrays/Temp/RGB/'+fileName+'.npy')
            flow = np.array([])
            training_RGB, targets_RGB, training_FLOW, targets_FLOW = get_rgb_and_flow(rgb, flow, binarizedTargets, features)
            training_X = np.concatenate((training_X, training_RGB), axis=0)
            training_Y = np.concatenate((training_Y, targets_RGB), axis=0)
        
            
        
        
#        training_X = np.concatenate((training_X, training_RGB), axis=0)
#        training_Y = np.concatenate((training_Y, targets_RGB), axis=0)
        
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
    print(training_X.shape)
    print(training_Y.shape)
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
        current_video = np.load('../../saved_numpy_arrays/RGB_as_numpy/'+ fileName + '.npy')

    if features == 'flow':
        current_video = np.load('../../saved_numpy_arrays/OpticalFlow_combined/of_'+fileName+'.npy')
    
    video_mat_file = sio.loadmat('../../SumMe/GT/'+ fileName +'.mat')
    video_targets = video_mat_file['gt_score']
    binarizedTargets = binarizeTargets(video_targets)
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

def getTestResults(videos, model, evals, features):
    for i in range(int(len(videos)*0.8), len(videos)):
        fileName_only = videos[i].split('/')[-1].split('.')[0]
        print(fileName_only)
        testing_X, testing_Y = getTestingdata(fileName_only, features)
#        testing_X = testing_X / 255.0
        print('got data ')
        targets = model.predict(testing_X) * 20.0
        np.save('../../saved_numpy_arrays/predictions/I3D/'+fileName_only+'_'+features+'.npy', targets)
        
        evals.evaluateSumMe(targets, fileName_only)

def load_chkpnt_model(features):
    model = create_model(features)
    model.load_weights("I3D_model/weights.hdf5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

feature = 'rgb' #'flow'
#model = train_model(feature)
model = load_models('I3D_model/Fine_tunned_model', feature)
#model = load_chkpnt_model( feature)

from SumMeEvaluation import SumMeEvaluation
evals = SumMeEvaluation()
getTestResults(videos, model, evals, feature)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def combine_flow_rgb():
    test_files_rgb = glob.glob('../../saved_numpy_arrays/predictions/I3D/*_rgb.npy')
    test_files_flow= glob.glob('../../saved_numpy_arrays/predictions/I3D/*_flow.npy')
    test_files_flow.sort()
    test_files_rgb.sort()
    rgb = []
    flow = []
    comb = []
#    com_smooth = []
    for pred in range(len(test_files_rgb)):
        rgb_preds = np.load(test_files_rgb[pred])
        flow_preds = np.load(test_files_flow[pred])
        print(test_files_rgb[pred])
        combined = (rgb_preds + flow_preds) / 2.0
#        predictions1 = smooth(np.squeeze(combined, axis=-1), 10)
        fileName_only = test_files_rgb[pred].split('/')[-1].split('.')[0].split('_rgb')[0]
#        print('rgb: ')
        r, r_mean = evals.evaluateSumMe(rgb_preds, fileName_only)
#        print('flow: ')
        f, f_mean = evals.evaluateSumMe(flow_preds, fileName_only)
#        print('combined: ')
        c, c_mean = evals.evaluateSumMe(combined, fileName_only)
#        print('combined smooth: ')
#        c_s, c_s_mean = evals.evaluateSumMe(predictions1, fileName_only)
        
        rgb.append(r)
        flow.append(f)
        comb.append(c)
#        com_smooth.append(c_s)
        
    print('rgb: ', np.mean(rgb))
    print('flow: ', np.mean(flow))
    print('comb: ', np.mean(comb))
#    print('com_smooth: ', np.mean(com_smooth))
        
        
combine_flow_rgb()
