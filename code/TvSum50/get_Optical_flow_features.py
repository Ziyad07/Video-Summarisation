

import cv2
import numpy as np

def perform_of(v):
    v = v.astype('uint8')
    f, r, c, d = v.shape
    print v[0].shape
    previous_frame = cv2.cvtColor(v[0], cv2.COLOR_BGR2GRAY)
    
    flows = []
    combinedFlow = []
    for i in range(1, f):
        current_frame = cv2.cvtColor(v[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, 0.5, 3, 15, 3, 5, 1.2, 0)

        xflow = flow[..., 0]
        yflow = flow[..., 1]
#        comb_flow = xflow + yflow
#        
#        combinedFlow.append(comb_flow)                
#        flows.append(xflow)
#        flows.append(yflow)
        
        flows.append(flow)
        previous_frame = current_frame

    return np.array(flows)#, np.array(combinedFlow)
    
    
import ReadFiletoNumpy2 as rfn
objectNumpy = rfn.ReadFileToNumpy()
#vidfileasnumpy = objectNumpy.vid2npy3(fileName)
#vid = vidfileasnumpy[0]
##cap = cv2.VideoCapture(fileName)
#x_y_flows = perform_of(vid)

import glob
file_names = glob.glob('../../Webscope_I4/ydata-tvsum50-v1_1/video/*.mp4')

for i in range(len(file_names)):

    new_fileName = file_names[i].split('/')[-1].split('.')[0]        
    print 'Computing: ', new_fileName
    vidfileasnumpy, frameCount = objectNumpy.vid2npy3(file_names[i])
    print vidfileasnumpy.shape
    print frameCount
    vid = vidfileasnumpy
    x_y_flows = perform_of(vid)

    np.save('../../saved_numpy_arrays/TvSum50/OpticalFlow/of_' + new_fileName + '.npy', x_y_flows)






