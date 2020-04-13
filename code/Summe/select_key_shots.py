import numpy as np
import glob
from SumMeEvaluation import SumMeEvaluation
from kts.cpd_auto import cpd_auto
import pdb
import scipy.io as sio

evals = SumMeEvaluation()

features_used = 'flow'
features = glob.glob('../../saved_numpy_arrays/I3D_features/FLOW/features/*.npy')
features.sort()

mat_files_path = '../../SumMe/GT/'

preds_path = '../../saved_numpy_arrays/predictions/I3D/'
add = '*_'+features_used+'.npy'
files = glob.glob(preds_path+add)
#single_file = np.load(files[0])
#window_length = 20
#m = 150

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
#        if any(c > 0.6 for c in preds[cps[i]:cps[i+1]]):
#        targets[cps[i]:cps[i+1]] = preds[cps[i]:cps[i+1]] # amend to include from frames - 0
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
        final_targets[cps_point_1:cps_point_2] = np.expand_dims(number_array, axis=-1)
        count=count+interval_len
#    for ind, value in enumerate(idx):
##        final_targets[cps[idx[j-1]]:cps[idx[j]]] = j # check this line properly
#        list_item_idx = cps[idx[ind]]
#        list_item_cps_idx = np.where(cps==list_item_idx)[0][0]
#        cps_point_1 = cps[list_item_cps_idx]
#        cps_point_2 = cps[list_item_cps_idx-1]
#        interval_len=cps_point_1-cps_point_2
#        if interval_len > 0:    
#            final_targets[cps_point_2:cps_point_1] = range(num_frames)[cps_point_2:cps_point_1]
        
#    pdb.set_trace()
    return final_targets, ks_targets

def get_cps_number(number_of_frames, fps):
    lengh = int(number_of_frames / int(fps))
    number_cps = int(lengh/(5))
    
    return number_cps

def test_cps(model):
    for i in range(int(len(features)*0.8), len(features)):
        current_file_x = np.load(features[i])
        fileName = features[i].split('/')[-1].split('.')[0]
        current_fps = sio.loadmat(mat_files_path+fileName+'.mat').get('FPS')[0][0]
        
        normalised_features = np.array([power_norm(e) for e in current_file_x])
        K = np.dot(normalised_features, normalised_features.T)
        
        m = get_cps_number(current_file_x.shape[0], current_fps) # avergae 5sec cps
        
        cps, scores = cpd_auto(K, m, 1)
        preds = model.predict(normalised_features)

        print(fileName)
        targets = create_key_shots(preds, cps)
        evals.evaluateSumMe(targets, fileName)


def test():
    for i in range(int(len(features)*0.8), len(features)):
        current_file_x = np.load(features[i])
        fileName = features[i].split('/')[-1].split('.')[0]
        current_fps = sio.loadmat(mat_files_path+fileName+'.mat').get('FPS')[0][0]
        
        normalised_features = np.array([power_norm(e) for e in current_file_x])
        K = np.dot(normalised_features, normalised_features.T)
        
        m = get_cps_number(current_file_x.shape[0], current_fps) # avergae 5sec cps
        
        cps, scores = cpd_auto(K, m, 1)
        preds = np.load(preds_path+fileName+'_'+features_used+'.npy')
#        preds_rand = np.random.random(preds.shape)
#        pdb.set_trace()
        print(fileName)
        targets = create_key_shots(preds, cps)
#        targets2 = create_key_shots(preds_rand, cps)
        evals.evaluateSumMe(targets, fileName)
#        evals.evaluateSumMe(targets2, fileName)
#        evals.evaluateSumMe(preds, fileName)
        
#        return targets
#    return fileName, cps, preds

#targets = test_cps()



