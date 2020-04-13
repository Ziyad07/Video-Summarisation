import scipy.io as sio
import numpy as np

class SumMeEvaluation(object):

    def __init__(self):
        pass

    def evaluateSumMe(self, predictions, videoName):
        HOMEDATA = '../../SumMe/GT/'        
        gt_file=HOMEDATA+'/'+videoName+'.mat'
        gt_data = sio.loadmat(gt_file)
        nFrames=gt_data.get('nFrames')
        
        nFrames = nFrames[0][0]    
        
        '''Example summary vector''' 
        #selected frames set to n (where n is the rank of selection) and the rest to 0
        summary_selections={};
#        a = np.random.random((nFrames,1))*20;
        summary_selections[0]=predictions
        summary_selections[0]=map(lambda q: (round(q) if (q >= np.percentile(summary_selections[0],85)) else 0),summary_selections[0])
        
        '''Evaluate'''
        #get f-measure at 15% summary length
        [f_measure,summary_length]=self.getF1Score(gt_file, summary_selections[0])
        print('F-measure : %.3f at length %.2f' % (f_measure, summary_length))
        return a        
        
        
    def getF1Score(self, gt_file, summary_selection):
        
        gt_data = sio.loadmat(gt_file)
         
        user_score=gt_data.get('gt_score')
        nFrames=user_score.shape[0];
        nbOfUsers=user_score.shape[1];
         
         # Compute pairwise f-measure, summary length and recall
        summary_indicator=np.array(map(lambda x: (1 if x>0 else 0),summary_selection)); 
        user_intersection=np.zeros((nbOfUsers,1));
        user_union=np.zeros((nbOfUsers,1));
        user_length=np.zeros((nbOfUsers,1));

        # The line below makes the binary codes for user scores (gt)- no matter the actual score
        gt_indicator=np.array(map(lambda x: (1 if x>0 else 0),user_score))
        user_intersection=np.sum(gt_indicator*summary_indicator);
        user_union=sum(np.array(map(lambda x: (1 if x>0 else 0),gt_indicator + summary_indicator)));         
        user_length=sum(gt_indicator)
        
        # The recall and precision is calculated per user    
        recall=float(user_intersection)/float(user_length);
        p=float(user_intersection)/float(np.sum(summary_indicator));

        print user_intersection, user_length
        
        f_measure=[]
#        for idx in range(0,len(p)):
        if p>0 or recall>0:
             f_measure.append(2*recall*p/(recall+p))
        else:
             f_measure.append(0)
        nn_f_meas=np.max(f_measure);
        f_measure=np.mean(f_measure);
        
        nnz_idx=np.nonzero(summary_selection)
        nbNNZ=len(nnz_idx[0])
             
        summary_length=float(nbNNZ)/float(len(summary_selection));
         
        return f_measure, summary_length
        
        
#summeEval = SumMeEvaluation()
#a = summeEval.evaluateSumMe('Cooking', '')