from pandas import DataFrame
import numpy as np
import glob


videoData = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_features/*.npy')
videoData_targets = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/*.npy')
df = DataFrame.from_csv("../../Webscope_I4/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno_cols.tsv", sep="\t")
feature_length = 4096 # c3d feature length
videoData.sort()
videoData_targets.sort()


videoName_Category = {}
for i in xrange(0, df.shape[0], 20):
    videoName_Category[df.index[i]] = df['Category'][i]
    
def readData():
    feature_matrix = np.zeros((1, feature_length))
    
    for i in range(len(videoData)):
        numpy_fileName = videoData[i]
#        print numpy_fileName
        fileName = numpy_fileName.split('/')[-1].split('.')[0]
        print fileName
        videoFile_category = videoName_Category[fileName]
        current_numpy_file = np.load(videoData[i])
        feature_matrix = np.vstack((feature_matrix, current_numpy_file))
        
        
    return feature_matrix[1:]
    
#feature_matrix = readData()
#
#np.save('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/x_train.npy', feature_matrix)

        
import shutil
testing_files = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_features/testing/*.npy')   
def getTestingTargets():
    for i in range(len(testing_files)):
        fileName = testing_files[i].split('/')[-1].split('.')[0]
#        print fileName
#        category = videoName_Category.get(fileName)
        new_file_name = fileName  + '.npy'
        print new_file_name
        shutil.move('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/'+new_file_name, '../../saved_numpy_arrays/TvSum50/RGB_as_numpy/testing/'+new_file_name)

getTestingTargets()


def prepareTrainingTargets():
    targets = np.zeros((1,))
    
    for i in range(len(videoData_targets)):
        averageList=[]
        current_video_targets = np.load(videoData_targets[i])
        partitions = current_video_targets.shape[0]/16
        frames = partitions * 16
        for j in xrange(0, frames, 16):
            average = 0
            for k in range(16):
                average = average + current_video_targets[j+k][0]
            average = np.round((average / 16), 2)
            averageList.append(average)
            
        targets = np.hstack((targets, averageList))
        
    return targets[1:]
        
#training_targets = prepareTrainingTargets()
#
#np.save('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/y_train', training_targets)

videoData_testing = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_features/testing/*.npy')
videoData_targets_testing = glob.glob('../../saved_numpy_arrays/TvSum50/ground_truth/testing/*.npy')
videoData_testing.sort()
videoData_targets_testing.sort()

def prepareTestingTargets():
    targets = np.zeros((1,))
    
    for i in range(len(videoData_targets_testing)):
        averageList=[]
        current_video_targets = np.load(videoData_targets_testing[i])
        partitions = current_video_targets.shape[0]/16
        frames = partitions * 16
        for j in xrange(0, frames, 16):
            average = 0
            for k in range(16):
                average = average + current_video_targets[j+k][0]
            average = np.round((average / 16), 2)
            averageList.append(average)
            
        targets = np.hstack((targets, averageList))
    
    np.save('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/y_test', targets[1:])
    return targets[1:]


def prepareTestData():
    feature_matrix = np.zeros((1, feature_length))
    
    for i in range(len(videoData_testing)):
        numpy_fileName = videoData_testing[i]
#        print numpy_fileName
        fileName = numpy_fileName.split('/')[-1].split('.')[0]
        print fileName
        current_numpy_file = np.load(videoData_testing[i])
        feature_matrix = np.vstack((feature_matrix, current_numpy_file))
    
    
    np.save('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/x_test.npy', feature_matrix[1:])
    return feature_matrix[1:]
    
    
#test_targets = prepareTestingTargets()
#test_feature_matrix = prepareTestData()













