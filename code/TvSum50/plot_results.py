import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import unicodedata
import pdb
import cv2
import matplotlib

def makeArraysEqual(gt_file, pred_file):
    new_array = np.zeros([len(gt_file), 1])
    
    for i in range(len(pred_file)):
        new_array[i] = pred_file[i]
        
    return new_array

plot1_good = '4wU_LUjG5Ic'
plot2_bad = 'EYqVtI9YWJA'

def read_video(fileName):
    path = '../../saved_numpy_arrays/TvSum50/RGB_as_numpy/testing/'
    video_file = np.load(path + fileName + '.npy')
    
    return video_file

def getSummary_frames(video_file, working_preds):
    summary_frames = []
    spacing = int(working_preds.shape[0] / 7.)
    j = 0
    for i in range(working_preds.shape[0]-2):
        if working_preds[i] == 1:
            h = j*spacing
            summary_frames.append(video_file[i+h])
            j=j+1
            
        if len(summary_frames) > 6:
            break

    return summary_frames
            
def splot_summary_frames(summary_frames, gt_name):
    fig, (ax_1, ax_2, ax_3, ax_4, ax_5, ax_6) = plt.subplots(2, 3, sharey=True, sharex=True)
#    fig.suptitle('Summary frames for video: '+ gt_name)

    ax_1.axis("off"); ax_2.axis("off"); ax_3.axis("off"); ax_4.axis("off"); ax_5.axis("off"); ax_6.axis("off");
    ax_1.imshow(cv2.cvtColor(summary_frames[0], cv2.COLOR_BGR2RGB))
    ax_2.imshow(cv2.cvtColor(summary_frames[1], cv2.COLOR_BGR2RGB))
    ax_3.imshow(cv2.cvtColor(summary_frames[2], cv2.COLOR_BGR2RGB))
    ax_4.imshow(cv2.cvtColor(summary_frames[3], cv2.COLOR_BGR2RGB))
    ax_5.imshow(cv2.cvtColor(summary_frames[4], cv2.COLOR_BGR2RGB))
    ax_6.imshow(cv2.cvtColor(summary_frames[5], cv2.COLOR_BGR2RGB))
    
    return fig

def plot_summary_frames(summary_frames, gt_name):
        
    fig = plt.figure(figsize=(7, 7))
    columns = 3
    rows = 2
    ax = []

    for i in range(columns*rows):
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i+1) )
        #ax[-1].set_title("ax:"+str(i))  # set title
        plt.imshow(cv2.cvtColor(summary_frames[i], cv2.COLOR_BGR2RGB))
        ax[-1].axis("off");

    plt.show() 
    
    return fig

def saveStartEndofSnippet(video_file, predictions, typeSummary):
    
    matplotlib.image.imsave('Plots/'+typeSummary+'_frame_0.jpg', cv2.cvtColor(video_file[0], cv2.COLOR_BGR2RGB))
    matplotlib.image.imsave('Plots/'+typeSummary+'_frame_n.jpg', cv2.cvtColor(video_file[-1], cv2.COLOR_BGR2RGB))
    # save frame at frame 0
    # save frame at frame n
#    pdb.set_trace()
    for i in range(predictions.shape[0]-2):
        
        if predictions[i] == 0 and predictions[i+1] == 1 and predictions[i+2] == 1:
            # save image at i+1
            matplotlib.image.imsave('Plots/'+typeSummary+'_frame_'+str(i+1)+'.jpg',  cv2.cvtColor(video_file[i+1], cv2.COLOR_BGR2RGB))
        if predictions[i] == 1 and predictions[i+1] == 1 and predictions[i+2] == 0:
            # save image at i+1
            matplotlib.image.imsave('Plots/'+typeSummary+'_frame_'+str(i+1)+'.jpg', cv2.cvtColor(video_file[i+1], cv2.COLOR_BGR2RGB))
        

def plot_results():
    predictionFiles = sio.loadmat('matlab_preds.mat').get('res')[0]
    ground_truth_path = '../../saved_numpy_arrays/TvSum50/ground_truth/testing/'
    
    for i in range(len(predictionFiles)):
        plots_dpi = 1000
        images_dpi = 500
        predictionFile = predictionFiles[i]
#    predictionFile = sio.loadmat('matlab_preds.mat').get('res')[0][video_number]
#        pdb.set_trace()
        gt_name = unicodedata.normalize('NFKD', predictionFile[1][0]).encode('ascii', 'ignore')
        gt_preds = predictionFile[0]
        gt_file = np.load(ground_truth_path + gt_name + '.npy')
        mean_result = predictionFile[5][0][0]
        mms = MinMaxScaler().fit_transform(gt_file)
        preds = makeArraysEqual(gt_file, gt_preds.T)
        new_preds = preds
        print(gt_name)
        print('Mean result: ', mean_result)
        print('Summary length: ', float(np.count_nonzero(new_preds))/ float(new_preds.shape[0]))
        if gt_name == plot1_good:
            
            video_file = read_video(gt_name)
            working_preds = new_preds[1000:3000]
            summary_frames = getSummary_frames(video_file, working_preds)
            
            saveStartEndofSnippet(video_file, new_preds, 'good')
            
            images = plot_summary_frames(summary_frames, gt_name)
            
            f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
            
            ax1.bar(np.arange(new_preds.shape[0]), np.squeeze(new_preds,axis=-1), 
                    align='center', alpha=0.5,
                    color='r')
            ax1.plot(mms)
            ax1.set_title('Ground truth and selected summary for video: '+ gt_name)
            ax1.set_ylabel('Normalised user scores')
            x = np.arange(1000, 3000, 1)
            ax2.plot(x,mms[1000:3000])
            ax2.bar(x, np.squeeze(new_preds[1000:3000],axis=-1), 
                    align='center', alpha=0.5,
                    color='r')
            ax2.set_xlabel('Frame number')
            f.savefig('Plots/TvSum_Plot_good.png', format='png', dpi=plots_dpi)
            images.savefig('Plots/TvSum_summframe_good.png', format='png', dpi=images_dpi)
            
        if gt_name == plot2_bad:
            
            video_file = read_video(gt_name)
            working_preds = new_preds[200:2200]
            summary_frames = getSummary_frames(video_file, working_preds)
            
            saveStartEndofSnippet(video_file, new_preds, 'bad')
            
            images = plot_summary_frames(summary_frames, gt_name)
            
            f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
            ax1.bar(np.arange(new_preds.shape[0]), np.squeeze(new_preds,axis=-1), 
                    align='center', alpha=0.5,
                    color='r')
            ax1.plot(mms)
            ax1.set_title('Ground truth and selected summary for video: '+ gt_name)
            ax1.set_ylabel('Normalised user scores')
            x = np.arange(200, 2200, 1)
            ax2.plot(x,mms[200:2200])
            ax2.bar(x, np.squeeze(new_preds[200:2200],axis=-1), 
                    align='center', alpha=0.5,
                    color='r')
            ax2.set_xlabel('Frame number')
            f.savefig('Plots/TvSum_Plot_bad.png', format='png', dpi=plots_dpi)
            images.savefig('Plots/TvSum_summframe_bad.png', format='png', dpi=images_dpi)
            
plot_results()
