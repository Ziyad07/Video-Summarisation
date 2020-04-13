from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import glob

model = ResNet50(weights='imagenet', include_top=False, pooling='max')
file_path = '../../saved_numpy_arrays/RGB_as_numpy/224_dims/*.npy'
files = glob.glob(file_path)
saved_path = '../../saved_numpy_arrays/Resnet_features/'
files.sort()

for i in range(len(files)):
    file_n = np.load(files[i])
    fileName = files[i].split('/')[-1].split('.')[0]
    
    x = preprocess_input(file_n)
    preds = model.predict(x)
    np.save(saved_path+fileName+'.npy', preds)
    print(fileName)
#preds = np.squeeze(np.squeeze(np.squeeze(preds, axis=0),axis=0), axis=0)