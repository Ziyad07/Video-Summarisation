from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
import numpy as np
from sklearn.metrics import classification_report
#import math
from Prepare_Data_TvSum import Prepare_Data

get_data = Prepare_Data()

# read data
x_train, y_train, x_test, y_test = get_data.read_c3d_hog_data()
#x_train, y_train, x_test, y_test = get_data.read_data()  # Just c3d features

#x_train = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/x_train.npy')
#x_test = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/x_test.npy')
#y_train = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/y_train.npy')
#y_test = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/training_data/y_test.npy')

input_dims = x_train.shape[1]
batchSize = 32


#y_train = np.array(map(lambda q: (math.ceil(q) if (q >= 2) else 0), y_train))
#y_test = np.array(map(lambda w: (math.ceil(w) if (w >= 2) else 0), y_test))

y_train = np.array(map(lambda q: (1 if (q >= 2) else 0), y_train))
y_test = np.array(map(lambda w: (1 if (w >= 2) else 0), y_test))


def computational_graph():
    model = Sequential()
    model.add(Dense(input_dims, input_dim=input_dims, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def eval_model_1(training_X, training_Y, testing_X, testing_Y):
    # Compile model
    model = computational_graph()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(training_X, training_Y, epochs=2, batch_size=batchSize)
    # evaluate the model
    scores = model.evaluate(testing_X, testing_Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # predict

    predictions = model.predict(testing_X)
    rounded_preds = [round(x[0]) for x in predictions]
    print classification_report(testing_Y, rounded_preds, target_names=['Not Summary', 'Summary'])

    return model


def expandTargetsVector(partitioned_targets):
    expanded_targets = []
    for snippet in range(len(partitioned_targets)):
        if partitioned_targets[snippet] > 0:
            for j in range(16):
                expanded_targets.append(partitioned_targets[snippet])
        else:
            for j in range(16):
                expanded_targets.append(partitioned_targets[snippet])

    return expanded_targets

#from sklearn.preprocessing import MinMaxScaler
#y_train = np.expand_dims(y_train, axis=-1)
#y_test = np.expand_dims(y_test, axis=-1)
#
#mms_targets = MinMaxScaler().fit(y_train)
#trainingY_scaled = mms_targets.transform(y_train)
#testingY_scaled = mms_targets.transform(y_test)
#
##trainingY_scaled  = training_Y * 100
##testingY_scaled  = testing_Y * 100
#
#trainingY_scaled = np.squeeze(trainingY_scaled, axis=-1)
#testingY_scaled = np.squeeze(testingY_scaled, axis=-1)


#model = eval_model_1(x_train, trainingY_scaled, x_test, testingY_scaled)
model = eval_model_1(x_train, y_train, x_test, y_test)


import glob
import scipy.io
testing_files = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_hog/testing/*.npy')
#testing_files = glob.glob('../../saved_numpy_arrays/TvSum50/c3d_features/testing/*.npy')
#testing_video = np.load('../../saved_numpy_arrays/TvSum50/c3d_features/testing/4wU_LUjG5Ic.npy')
#testing_targets = np.load('../../saved_numpy_arrays/TvSum50/average_targets/testing/4wU_LUjG5Ic PR.npy')
##

for i in range(len(testing_files)):
    current_testing_file = testing_files[i]
    fileName = current_testing_file.split('/')[-1].split('.')[0]
    
    numpy_feature_matrix = np.load(current_testing_file)    
    
    preds = model.predict(numpy_feature_matrix)
    full_preds = expandTargetsVector(preds)
    print fileName
    scipy.io.savemat('../../saved_numpy_arrays/TvSum50/predictions/simpleNN/c3d_hog/' + fileName + '.mat', dict(x=full_preds))
#    scipy.io.savemat('../../saved_numpy_arrays/TvSum50/predictions/simpleNN/' + fileName + '.mat', dict([('preds', preds)]))
#    np.save('../../saved_numpy_arrays/TvSum50/predictions/simpleNN/' + fileName)







































