import numpy as np
import scipy
from summe import *

class evaluate_metrics():

    def __init__(self):
        pass

    def getSumMeEvaluationMetrics(self, targets, videoName):
        HOMEDATA = '../SumMe/GT';

        # In this example we need to do this to now how long the summary selection needs to be
        gt_file = HOMEDATA + '/' + videoName + '.mat'
        gt_data = scipy.io.loadmat(gt_file)
        nFrames = gt_data.get('nFrames')

        nFrames = nFrames[0][0]
        full_frame_targets = np.zeros((nFrames, 1))
        for i in range(len(targets)):
            full_frame_targets[i] = targets[i]

        '''Example summary vector'''
        # selected frames set to n (where n is the rank of selection) and the rest to 0
        summary_selections = {};
        summary_selections[0] = full_frame_targets
        summary_selections[0] = map(lambda q: (round(q) if (q >= np.percentile(summary_selections[0], 85)) else 0),
                                    summary_selections[0])

#        print summary_selections[0]
        '''Evaluate'''
        # get f-measure at 15% summary length
        [f_measure, summary_length] = evaluateSummary(summary_selections[0], videoName, HOMEDATA)
        print('F-measure : %.3f at length %.2f' % (f_measure, summary_length))

        '''plotting'''
    #    methodNames={'Neural Network'};
    #    plotAllResults(summary_selections,methodNames,videoName,HOMEDATA);
