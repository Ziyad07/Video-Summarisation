# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:53:19 2018

@author: ziyad
"""
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

from Prepare_Data import Prepare_Data
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from evaluate_metrics import evaluate_metrics

get_data = Prepare_Data()
# read data
# training_X, training_Y, testing_X, testing_Y = get_data.read_c3d_hog_data()
training_X, training_Y, testing_X, testing_Y = get_data.read_data()  # Just c3d features

input_dims = training_X.shape[1]
batchSize = 32


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
    model.fit(training_X, training_Y, epochs=5, batch_size=batchSize)
    # evaluate the model
    scores = model.evaluate(testing_X, testing_Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # predict

    #    predictions = model.predict(testing_X)
    #    rounded_preds = [round(x[0]) for x in predictions]
    #    print classification_report(testing_Y, rounded_preds, target_names=['Not Summary', 'Summary'])

    return model


def summarizeVideo(model):
    listOfTestingVideos, listOftargets = get_data.get_training_data_mAP()
    for i in range(len(listOfTestingVideos)):
        predictions = model.predict(listOfTestingVideos[i][1])
        fileName = '../SumMe/videos/' + listOfTestingVideos[i][0]
        list_preds = [x[0] for x in predictions]
        expandedTargets = expandTargetsVector(list_preds)
        testingFile = listOftargets[i][0].split('.')[0]
        print testingFile
        evaluate_metric = evaluate_metrics()
        evaluate_metric.getSumMeEvaluationMetrics(expandedTargets, testingFile)

    return expandedTargets


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


model = eval_model_1(training_X, training_Y, testing_X, testing_Y)
targets = summarizeVideo(model)
