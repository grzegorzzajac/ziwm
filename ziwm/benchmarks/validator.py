#!/usr/bin/python2.7

import numpy as np
import time
from sklearn.preprocessing.data import OneHotEncoder

def __average_square_error(Y1, Y2):
    length = Y1.shape[0]
    assert Y1.shape == Y2.shape
    
    error = np.sum((Y1 - Y2) ** 2.0)
    error = error / (1.0 * length)
    return error

def __regression_error(Y1, Y2):
    return __average_square_error(Y1, Y2)

def __classification_error(Y_predicted, Y):
    assert Y_predicted.shape == Y.shape
    length = Y_predicted.shape[0]

    Y_class = np.rint(Y).astype(int)
    Y_pred_class = np.rint(Y_predicted).astype(int)

    error = ((Y_pred_class != Y_class) * 1.).sum()
    error = error / float(length)
    return error

def model_score(model, X_test, Y_test, problem_type):
    '''
    Calculates performance score of a model on a given examples set
    '''
    Y_predicted = model.predict(X_test)
    assert Y_predicted.shape == Y_test.shape
    
    if model.name() == 'hpelmnn':
        Y_predicted = Y_predicted.argmax(axis=1)
        Y_test = Y_test.argmax(axis=1)

    if problem_type == 'regression':
        score = 1.0 - __regression_error(Y_predicted, Y_test)
    elif problem_type == 'classification':
        score = 1.0 - __classification_error(Y_predicted, Y_test)
    else:
        raise Exception("problem_type not recognized", problem_type)
    
    return score

def model_score_kfold(model, X, Y, kfold_labels, problem_type, measure_time=False):
    kfold_iterations = len(kfold_labels)
    score_sum = 0.0
    Y = np.rint(Y).astype(int)
    if measure_time == True:
        total_time = 0.0
    for train_index, test_index in kfold_labels:
        if model.name() == 'hpelmnn':
            Y_arr = Y.reshape(-1, 1)
            enc = OneHotEncoder()
            enc.fit(Y_arr)
            Y_OHE = enc.transform(Y_arr).toarray()
            Y_train, Y_test = Y_OHE[train_index, :], Y_OHE[test_index, :]
        else:
            Y_train, Y_test = Y[train_index], Y[test_index]
        X_train, X_test = X[train_index], X[test_index]
            
        if measure_time == True:
            t1 = time.time()
        model.train(X_train, Y_train)
        if measure_time == True:
            t2 = time.time()
            time_diff = (t2 - t1) * 1000.0
            total_time += time_diff

        score = model_score(model, X_test, Y_test, problem_type)
        score_sum += score
        
    score = score_sum / float(kfold_iterations)
    if measure_time:
        return score, total_time
    else:
        return score
