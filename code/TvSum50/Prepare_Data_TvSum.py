# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:50:05 2018

@author: ziyad

"""

import numpy as np
import glob 


class Prepare_Data(object):

    def __init__(self):    
        self.feature_vector_names = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_features/*.npy')
        self.target_names = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/*.npy')
        self.hog_features = glob.glob('../../saved_numpy_arrays/TvSum50/HOG_Features/*.npy')
        self.hog_c3d_features = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_hog/*.npy')

        self.feature_vector_names_test = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_features/testing/*.npy')
        self.target_names_test = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/testing/*.npy')
        self.hog_features_test = glob.glob('../../saved_numpy_arrays/TvSum50/HOG_Features/testing/*.npy')
        self.hog_c3d_features_test = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_hog/testing/*.npy')    
        
        self.targetPath = '../../saved_numpy_arrays/TvSum50/ground_truth/'
        self.hog_featuresPath = '../../saved_numpy_arrays/TvSum50/HOG_Features/'

        self.targetPath_test = '../../saved_numpy_arrays/TvSum50/ground_truth/testing/'
        self.hog_featuresPath_test = '../../saved_numpy_arrays/TvSum50/HOG_Features/testing/'
        
#        self.feature_vector_names_test.sort()
#        self.target_names.sort()
#        self.hog_features.sort()
#        self.hog_c3d_features.sort()
#        self.feature_vector_names_test.sort()
#        self.target_names_test.sort()
#        self.hog_features_test.sort()
#        self.hog_c3d_features_test.sort()
    
    def prepare_train_data(self):
        final_feature_matrix = np.zeros([7232])
        final_targets = []
        
        for i in range(len(self.feature_vector_names)): # 80% for training
            fileInProgress = self.feature_vector_names[i].split('/')[-1].split('.')[0]
            print('Computing: ', fileInProgress)
            feature_matrix = np.load(self.feature_vector_names[i])
            category = videoName_Category.get(fileInProgress)
            print category
            targets = np.load(self.targetPath + fileInProgress + ' ' + category + '.npy')            
            targets = self.prepareTrainingTargets(targets)
            hog_feats = np.load(self.hog_featuresPath + fileInProgress + '.npy')
#            targets = np.squeeze(targets, axis = -1)
            c3d_hog_feature_matrix = np.concatenate((feature_matrix, hog_feats), axis=1)        
            
            final_feature_matrix = np.vstack((final_feature_matrix, c3d_hog_feature_matrix))
            final_targets = np.hstack((final_targets, targets))
            
            np.save('../../saved_numpy_arrays/TvSum50/c3d_hog/'+ fileInProgress + '.npy', c3d_hog_feature_matrix)
            
        return final_feature_matrix, final_targets
        
        
    def prepare_test_data(self):
        final_feature_matrix = np.zeros([7232]) #for hog and c3d --- make it 4096 just for c3d
        final_targets = []
        sums = 0
        for i in range(len(self.feature_vector_names_test)): 
            fileInProgress = self.feature_vector_names_test[i].split('/')[-1].split('.')[0]
            print('Computing Testing : ', fileInProgress)
            feature_matrix = np.load(self.feature_vector_names_test[i])
            category = videoName_Category.get(fileInProgress)
            targets = np.load(self.targetPath_test + fileInProgress + ' ' + category + '.npy')
            targets = self.prepareTrainingTargets(targets)
            print targets.shape
            sums = sums + targets.shape[0]
#            targets = np.squeeze(targets, axis = -1)
            hog_feats = np.load(self.hog_featuresPath_test + fileInProgress + '.npy')
            
            c3d_hog_feature_matrix = np.concatenate((feature_matrix, hog_feats), axis=1)                                
            print c3d_hog_feature_matrix.shape
            final_feature_matrix = np.vstack((final_feature_matrix, c3d_hog_feature_matrix))            
            final_targets = np.hstack((final_targets, targets))
            
#            np.save('../../saved_numpy_arrays/TvSum50/c3d_hog/testing/'+ fileInProgress + '.npy', c3d_hog_feature_matrix)            
        print sums
        return final_feature_matrix, final_targets   


    def save_data_numpy(self):
    
        training_X, training_Y = self.prepare_train_data()
        testing_X, testing_Y = self.prepare_test_data()
        
        np.save('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/training_X.npy', training_X)
        np.save('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/training_Y.npy', training_Y)
        np.save('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/testing_X.npy', testing_X)
        np.save('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/testing_Y.npy', testing_Y)
    
    
    def read_data(self):
        training_X = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/x_train.npy')
        training_Y = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/y_train.npy')
        testing_X = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/x_test.npy')
        testing_Y = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/y_test.npy')
        
        return training_X, training_Y, testing_X, testing_Y   # get rid of the zero row we appended initially
        
        
        
    def read_c3d_hog_data(self):
        training_X = np.load('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/training_X.npy')
        training_Y = np.load('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/training_Y.npy')
        testing_X = np.load('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/testing_X.npy')
        testing_Y = np.load('../../saved_numpy_arrays/TvSum50/c3d_hog/combined_data/testing_Y.npy')
        
        return training_X[1:], training_Y, testing_X[1:], testing_Y   # get rid of the zero row we appended initially
    
        
            
    def prepareTrainingTargets(self, current_video_targets):
        averageList=[]
        partitions = current_video_targets.shape[0]/16
        frames = partitions * 16
        for j in xrange(0, frames, 16):
            average = 0
            for k in range(16):
                average = average + current_video_targets[j+k][0]
            average = np.round((average / 16), 2)
            averageList.append(average)
            
        targets = np.array(averageList)
            
        return targets
                
        
        

#from pandas import DataFrame
#df = DataFrame.from_csv("../../Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno_cols.tsv", sep="\t")
#videoName_Category = {}
#for i in xrange(0, df.shape[0], 20):
#    videoName_Category[df.index[i]] = df['Category'][i]
#    
#objectIns = Prepare_Data()

#x_train, y_train, x_test, y_test = objectIns.read_c3d_hog_data()
#objectIns.save_data_numpy()
#objectIns.prepare_train_data()
#x_test, y_test = objectIns.prepare_test_data() 
#listOfTestingVideos, listOftargets = objectIns.get_training_data_mAP()
        
        
        
        
        
        
        
        
        
        
        