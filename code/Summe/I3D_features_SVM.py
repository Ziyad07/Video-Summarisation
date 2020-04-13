import numpy as np
import glob
from sklearn import svm

feature = 'RGB'
path = '../../saved_numpy_arrays/I3D_features/' + feature + '/'
files = glob.glob(path+'features/*.npy')
files.sort()

def read_features():    
    features = np.zeros([1, 1024])
    targets = np.array([])
    for i in range(int(len(files)*0.8)):
        current_file_features = files[i]
        fileName = current_file_features.split('/')[-1].split('.')[0]
        numpy_file_features = np.load(current_file_features)
        numpy_file_targets = np.load(path+'targets/'+fileName+'.npy')
        features = np.concatenate((features, numpy_file_features), axis=0)
        targets = np.concatenate((targets, numpy_file_targets), axis=0)
        print(fileName)
        
    print(targets.shape)
    print(features[1:].shape)
    return features[1:], targets

def power_norm(v):
    normalised_vector = np.sign(v) * np.sqrt(np.abs(v))
    return normalised_vector

def train_SVM(features, targets):
    model = svm.LinearSVC(random_state=0, verbose=1)
    model.fit(features, targets)
    return model

def test(model, evals, features):
    for i in range(int(len(files)*0.8), len(files)):
        current_file_features = files[i]
        fileName = current_file_features.split('/')[-1].split('.')[0]
        print(fileName)
        numpy_file_features = np.load(current_file_features)
        normalised_features = np.array([power_norm(e) for e in numpy_file_features])
#        numpy_file_targets = np.load(path+'targets/'+fileName+'.npy')
#        score = model.score(numpy_file_features, numpy_file_targets)
#        print(score)
        predictions = model.predict(normalised_features) * 20.0
        np.save('../../saved_numpy_arrays/predictions/I3D/SVM/'+fileName+'_'+features+'.npy', predictions)
            
        evals.evaluateSumMe(predictions, fileName)
#        return predictions

def test_RGB_FLOW_comb(evals):
    svm_path = '../../saved_numpy_arrays/predictions/I3D/SVM/'
    flow_preds_files = glob.glob(svm_path+'*_RGB.npy')
    print(flow_preds_files)
    for i in range(len(flow_preds_files)):
        fileName = flow_preds_files[i].split('/')[-1].split('_RGB')[0]
        rgb_file = np.load(flow_preds_files[i])
        flow_file = np.load(svm_path + fileName + '_FLOW.npy')
        print(fileName)
        combined = (rgb_file + flow_file) / 2.0
        evals.evaluateSumMe(combined, fileName)
        
from SumMeEvaluation import SumMeEvaluation
evals = SumMeEvaluation()

features, targets = read_features()
normalize_features = np.array([power_norm(e) for e in features])
model = train_SVM(normalize_features, targets)

#test(model, evals, feature)
test_RGB_FLOW_comb(evals)

