import cv2
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import glob

files = glob.glob('../../saved_numpy_arrays/RGB_as_numpy/*.npy')

n = 10
def get_distribution(leftside, rightside):
    hist,bins = np.histogram(leftside,256,[0,256])    
    hist2,bins2 = np.histogram(rightside,256,[0,256])    
    
    return hist, hist2

#center_frame = int(np.ceil(file_npy.shape[0] / 2))


def loop_through_video(file_npy):
    means = np.zeros([file_npy.shape[0]])
    stds = np.zeros([file_npy.shape[0]])
    for i in range(n, len(file_npy)-n):
        left_ravel = get_left_raveled_images(i, file_npy)
        right_ravel = get_right_raveled_images(i, file_npy)
        l_mean, l_std = np.mean(left_ravel), np.std(left_ravel)
        r_mean, r_std = np.mean(right_ravel), np.std(right_ravel)
        means[i] = np.abs(l_mean - r_mean)
        stds[i] = np.abs(l_std - r_std)
        
    return means, stds
    
def get_left_raveled_images(center_frame, file_npy):
    left_ravel = np.zeros([37632])
    for i in range(center_frame - n, center_frame, 1):
        raveled_image = np.ravel(file_npy[i])
        left_ravel += raveled_image
    
    left_ravel = np.round(left_ravel / n)
    return left_ravel

def get_right_raveled_images(center_frame, file_npy):
    right_ravel = np.zeros([37632]) # 112*112*3
    for i in range(center_frame, center_frame + n, 1):
        raveled_image = np.ravel(file_npy[i])
        right_ravel += raveled_image
    
    right_ravel = np.round(right_ravel / n)
    return right_ravel

def normalise(X):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (X.max() - X.min()) + X.min()
    
    return X_scaled

def plot_results(gt_scores, means, stds, fileName):
    fig, axs = plt.subplots(3, 1)
    t = np.arange(0, gt_scores.shape[0], 1)
    plt.subplots_adjust(hspace=1)

    axs[0].plot(t, stds, label='stds')
    axs[0].plot(t, means, label='means')
    axs[0].set_xlabel('frames')
    axs[0].set_ylabel('abs residual')
    axs[0].legend(loc='upper left', shadow=True)
    
    axs[1].plot(t, gt_scores, label='gt_scores')
    axs[1].set_xlabel('frames')
    axs[1].set_ylabel('gt_score')
    axs[1].legend(loc='upper left', shadow=True)
        
    axs[2].plot(t, means, label='means')
    axs[2].plot(t, stds, label='stds')
    axs[2].plot(t, gt_scores, label='gt_scores')
    axs[2].set_xlabel('frames')
    axs[2].set_ylabel('gt_scres with residual')
    axs[2].legend(loc='upper left', shadow=True)
#    plt.gca().set_position([0, 0, 1, 1])
    from os.path import expanduser
    home = expanduser("~")
    
    fig.savefig(home+'/Pictures/'+fileName+'.png')

def get_plots_for_all_videos():
    for i in range(len(files)):
        fileName = files[i].split('/')[-1].split('.')[0]
        file_npy = np.load('../../saved_numpy_arrays/RGB_as_numpy/'+fileName+'.npy')
        means, stds = loop_through_video(file_npy)
        scaler = MinMaxScaler()
        means = scaler.fit_transform(means)
        stds = scaler.fit_transform(stds)
        print(fileName)
        gt_scores = sio.loadmat('../../SumMe/GT/'+fileName+'.mat').get('gt_score')
        plot_results(gt_scores, means, stds, fileName)
        
get_plots_for_all_videos()



#left_hist, right_hist = get_distribution(left, right)
#from scipy.stats import ks_2samp
#ks_2samp(left_hist, right_hist)
