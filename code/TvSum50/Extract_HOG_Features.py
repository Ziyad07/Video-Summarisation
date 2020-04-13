import cv2
from skimage.feature import hog
from Extract_C3D import Extract_C3D
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from ReadFileToNumpy2 import ReadFileToNumpy



file_names = glob.glob('../../Webscope_I4/ydata-tvsum50-v1_1/video/*.mp4')
extract = ReadFileToNumpy()

for i in range(len(file_names)):
    print i
    video_as_numpy, frameCount = extract.vid2npy3(file_names[i])
    feature_matrix = np.zeros([3136])
    frames = frameCount - (frameCount % 16)

    for index in tqdm(xrange(0, frames, 16)):
        img = Image.fromarray(video_as_numpy[index], 'RGB')
        img = np.asarray(img.convert('L'))
        hogHist, hogImage = hog(img, orientations=16, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
        reshape_hogHist = hogHist.reshape((1,-1))
        feature_matrix = np.vstack((feature_matrix, reshape_hogHist))
        
    new_fileName = file_names[i].split('/')[-1].split('.')[0]        
    np.save('../../saved_numpy_arrays/TvSum50/HOG_Features/' + new_fileName + '.npy', feature_matrix[1:])