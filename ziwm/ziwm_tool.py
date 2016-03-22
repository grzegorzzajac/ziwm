#!/bin/usr/python2

from os import path
import sys
sys.path.append(path.abspath('..'))

from ziwm.model.base import Model
from ziwm.data.dataset.base import Dataset
from ziwm.data.utils import split_dataset
from ziwm.validator import model_score

if __name__ == '__main__':
    '''
    Tests performance of prediction models on all available datasets
    '''

    # load all possible models and datasets
    datasets = Dataset.all_datasets()
    models = Model.all_models()
    
    # print output header
    print("dataset,model,score")

    for dataset in datasets:

        # load dataset
        X, Y = dataset.load()
        
        # split dataset into train and test sets
        X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
        
        # evaluate models
        for model in models:

            # train model
            model.train(X_train, Y_train)
            
            # calculate performance score
            score = model_score(model, X_test, Y_test, dataset.problem_type())
            
            # print results in csv format
            print("{0},{1},{2}".format(dataset.name(), model.name(), score))
            
