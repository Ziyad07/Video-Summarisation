

import cv2
import numpy as np

fileName = '../SumMe/videos/Paintball.mp4'
#cap = cv2.VideoCapture(fileName)
#
#ret, frame1 = cap.read()
#prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#hsv = np.zeros_like(frame1)
#hsv[...,1] = 255
#
#while(1):
#    ret, frame2 = cap.read()
#    nexts = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#
#    flow = cv2.calcOpticalFlowFarneback(prvs,nexts, 0.5, 3, 15, 3, 5, 1.2, 0)
#
#    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#    hsv[...,0] = ang*180/np.pi/2
#    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#
#    cv2.imshow('frame2',rgb)
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
#    elif k == ord('s'):
#        cv2.imwrite('opticalfb.png',frame2)
#        cv2.imwrite('opticalhsv.png',rgb)
#    prvs = nexts
#
#cap.release()
#cv2.destroyAllWindows()

def perform_of(v):
    v = v.astype('uint8')
    f, r, c, d = v.shape
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
    
    
import ReadFiletoNumpy as rfn
objectNumpy = rfn.ReadFileToNumpy()
#vidfileasnumpy = objectNumpy.vid2npy3(fileName)
#vid = vidfileasnumpy[0]
##cap = cv2.VideoCapture(fileName)
#x_y_flows = perform_of(vid)

import glob
file_names = glob.glob('../SumMe/videos/*.mp4')

for i in range(len(file_names)):

    new_fileName = file_names[i].split('/')[-1].split('.')[0]        
    print 'Computing: ', new_fileName
    vidfileasnumpy, frameCount = objectNumpy.vid2npy3(file_names[i])
    print frameCount
    vid = vidfileasnumpy
    x_y_flows = perform_of(vid)

    np.save('../saved_numpy_arrays/OpticalFlow_combined/of_' + new_fileName + '.npy', x_y_flows)






