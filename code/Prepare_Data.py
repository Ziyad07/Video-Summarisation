# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:50:05 2018

@author: ziyad

"""

import numpy as np
import glob 


class Prepare_Data(object):

    def __init__(self):    
        self.feature_vector_names = glob.glob('../saved_numpy_arrays/c3d_features/fv*')
        self.target_names = glob.glob('../saved_numpy_arrays/c3d_features/targets*')
        self.hog_features = glob.glob('../saved_numpy_arrays/hog_features/hog*')
        self.hog_c3d_features = glob.glob('../saved_numpy_arrays/c3d_hog_per_video/fv*')
        self.feature_vector_names.sort()
        self.target_names.sort()
        self.hog_features.sort()
        self.hog_c3d_features.sort()
    
    
    def prepare_train_data(self):
        final_feature_matrix = np.zeros([7232])
        final_targets = []
        
        for i in range(int(len(self.feature_vector_names) * 0.8)): # 80% for training
            fileInProgress = self.feature_vector_names[i].split('/')[-1]
            print('Computing: ', fileInProgress)
            feature_matrix = np.load(self.feature_vector_names[i])
            targets = np.load(self.target_names[i])            
            hog_feats = np.load(self.hog_features[i])

            c3d_hog_feature_matrix = np.concatenate((feature_matrix, hog_feats), axis=1)        
            
            final_feature_matrix = np.vstack((final_feature_matrix, c3d_hog_feature_matrix))
            final_targets = np.hstack((final_targets, targets))
            
            np.save('../saved_numpy_arrays/c3d_hog_per_video/'+ fileInProgress, c3d_hog_feature_matrix)
            
        return final_feature_matrix, final_targets 
        
        
    def prepare_test_data(self):
        final_feature_matrix = np.zeros([7232]) #for hog and c3d --- make it 4096 just for c3d
        final_targets = []
        
        for i in xrange(int(len(self.feature_vector_names) * 0.8), len(self.feature_vector_names), 1): # 80% for testing
            fileInProgress = self.feature_vector_names[i].split('/')[-1]
            print('Computing: ', fileInProgress)
            feature_matrix = np.load(self.feature_vector_names[i])
            targets = np.load(self.target_names[i])            
            hog_feats = np.load(self.hog_features[i])
            
            c3d_hog_feature_matrix = np.concatenate((feature_matrix, hog_feats), axis=1)                                
            
            final_feature_matrix = np.vstack((final_feature_matrix, c3d_hog_feature_matrix))            
            final_targets = np.hstack((final_targets, targets))
            
            np.save('../saved_numpy_arrays/c3d_hog_per_video/'+ fileInProgress, c3d_hog_feature_matrix)            
            
        return final_feature_matrix, final_targets   


    def save_data_numpy(self):
    
        training_X, training_Y = self.prepare_train_data()
        testing_X, testing_Y = self.prepare_test_data()
        
        np.save('../saved_numpy_arrays/c3d_hog__train_test_features/training_X.npy', training_X)
        np.save('../saved_numpy_arrays/c3d_hog__train_test_features/training_Y.npy', training_Y)
        np.save('../saved_numpy_arrays/c3d_hog__train_test_features/testing_X.npy', testing_X)
        np.save('../saved_numpy_arrays/c3d_hog__train_test_features/testing_Y.npy', testing_Y)
    
    
    def read_data(self):
        training_X = np.load('../saved_numpy_arrays/c3d_features/training_X.npy')
        training_Y = np.load('../saved_numpy_arrays/c3d_features/training_Y.npy')
        testing_X = np.load('../saved_numpy_arrays/c3d_features/testing_X.npy')
        testing_Y = np.load('../saved_numpy_arrays/c3d_features/testing_Y.npy')
        
        return training_X[1:], training_Y, testing_X[1:], testing_Y   # get rid of the zero row we appended initially
        
        
        
    def read_c3d_hog_data(self):
        training_X = np.load('../saved_numpy_arrays/c3d_hog__train_test_features/training_X.npy')
        training_Y = np.load('../saved_numpy_arrays/c3d_hog__train_test_features/training_Y.npy')
        testing_X = np.load('../saved_numpy_arrays/c3d_hog__train_test_features/testing_X.npy')
        testing_Y = np.load('../saved_numpy_arrays/c3d_hog__train_test_features/testing_Y.npy')
        
        return training_X[1:], training_Y, testing_X[1:], testing_Y   # get rid of the zero row we appended initially
        
    
    def get_testing_data_c3d_hog(self):
        testingDataAsList_perVideo = self.hog_c3d_features
        targetNames = self.target_names
        split_testing_data_perVideo = []
        split_targets_data_perVideo = []
        
        for video in xrange(len(testingDataAsList_perVideo) - 5, len(testingDataAsList_perVideo), 1):
            targets = np.load(targetNames[video])            
            video_fv = np.load(testingDataAsList_perVideo[video])
#            print testingDataAsList_perVideo[video]
            file_name = testingDataAsList_perVideo[video].split('.npy')[-2].split('/')[-1].split('fv_')[-1] + '.mp4'
            
            data_object_targets = []
            data_object_targets.insert(0, file_name)
            data_object_targets.insert(1, targets)
            
            data_object_video_fv = []
            data_object_video_fv.insert(0, file_name)
            data_object_video_fv.insert(1, video_fv)
            
            split_testing_data_perVideo.insert(video, data_object_video_fv)
            split_targets_data_perVideo.insert(video, data_object_targets)
        
        
        return split_testing_data_perVideo, split_targets_data_perVideo
        

    def get_training_data_mAP(self):
        testingDataAsList_perVideo = self.feature_vector_names
        targetNames = self.target_names
        split_testing_data_perVideo = []
        split_targets_data_perVideo = []
        
        for video in xrange(len(testingDataAsList_perVideo) - 5, len(testingDataAsList_perVideo), 1):
            targets = np.load(targetNames[video])            
            video_fv = np.load(testingDataAsList_perVideo[video])
            file_name = testingDataAsList_perVideo[video].split('.npy')[-2].split('/')[-1].split('fv_')[-1] + '.mp4'
            
            data_object_targets = []
            data_object_targets.insert(0, file_name)
            data_object_targets.insert(1, targets)
            
            data_object_video_fv = []
            data_object_video_fv.insert(0, file_name)
            data_object_video_fv.insert(1, video_fv)
            
            split_testing_data_perVideo.insert(video, data_object_video_fv)
            split_targets_data_perVideo.insert(video, data_object_targets)
        
        
        return split_testing_data_perVideo, split_targets_data_perVideo
        
        
        
objectIns = Prepare_Data()
objectIns.get_testing_data_c3d_hog()
#objectIns.prepare_train_data()
#objectIns.prepare_test_data() 
#listOfTestingVideos, listOftargets = objectIns.get_training_data_mAP()
        
        
        