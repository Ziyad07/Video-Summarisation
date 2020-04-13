import cv2
import numpy as np

class ReadFileToNumpy(object):

    def vid2npy3(self, fileName):
        cap = cv2.VideoCapture(fileName)
        videoFrameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        frameCount = videoFrameCount
        # frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        # frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frameHeight = 112
        frameWidth = 112
        buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
        fc = 0
        ret = True
        print 'here'
        while (fc < frameCount and ret):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray, (frameHeight, frameWidth))

            buf[fc] = resize
            fc += 1
        cap.release()
        return buf, frameCount