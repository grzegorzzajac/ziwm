#!/usr/bin/python2.7

from ziwm.model.base_classifier.classifier import Classifier
from ziwm.data.base import Dataset
from ziwm.data.utils import split_dataset
from ziwm.benchmarks.validator import model_score

import numpy as np
from ziwm.model.voting_system.voting_system import VotingSystem
from ziwm.model.ensemble.ensemble import Ensemble

if __name__ == '__main__':
    '''
    Tests performance of prediction models on all available datasets
    '''

    # load all possible models and datasets
    datasets = Dataset.all_datasets()
    models = Classifier.all_models()
    voting_systems = VotingSystem.all_voting_systems()
    ensemble_types = Ensemble.all_ensemble_types()
    
    output_string_format = "{0: <15}\t{1: <28}\t{2: <13}\t{3: <18}\t{4: <28}\t{5: <22}\t{6: <15}\t{7: <22}"

    # print output header
    print(output_string_format
          .format('score','dataset','dataset_size','number_of_classes','model','ensemble','ensemble_size','voting_system'))

    for dataset in datasets:

        # load mock_classifier
        X, Y = dataset.load()
        # dataset info
        dataset_size = X.shape[0]
        classes_count = np.unique(Y).size

        
        # split mock_classifier into train and examples sets
        X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
        
        # evaluate models
        for model in models:

            # train model
            model.train(X_train, Y_train)
            
            # calculate performance score
            score = model_score(model, X_test, Y_test, dataset.problem_type())
            
            # print results in csv format
            print(output_string_format
                  .format(score, dataset.name(), dataset_size\
                          ,classes_count, model.name()\
                          ,'NONE', 'NONE', 'NONE'))

            # score enembles based on current model
            for voting_system in voting_systems:
                
                for ensemble_type in ensemble_types:
                    
                    # create ensemble
                    classifiers_in_ensamble = 5
                    ensemble = ensemble_type(voting_system, type(model), classifiers_in_ensamble)
                    ensemble.train(X_train, Y_train)
                    
                    # calculate performance score of ensemble
                    score = model_score(ensemble, X_test, Y_test, dataset.problem_type())
            
                    # print results in csv format
                    print(output_string_format
                          .format(score, dataset.name(), dataset_size\
                                  ,classes_count, model.name()\
                                  ,ensemble.name(), classifiers_in_ensamble\
                                  ,voting_system.name()))
