#!/usr/bin/python2.7

from ziwm.model.base_classifier.classifier import Classifier
from ziwm.benchmarks.validator import model_score_kfold

import numpy as np
from ziwm.model.voting_system.voting_system import VotingSystem
from ziwm.model.ensemble.ensemble import Ensemble
from ziwm.data.dataset_loader.dataset_loader import DatasetLoader
from sklearn.cross_validation import StratifiedKFold

if __name__ == '__main__':
    '''
    Tests performance of prediction models on all available datasets
    '''

    # load all possible models and datasets
    dataset_loader = DatasetLoader()
    dataset_loader.load()
    datasets = dataset_loader.get_all_datesets()
    models = Classifier.all_models()
    voting_systems = VotingSystem.all_voting_systems()
    ensemble_types = Ensemble.all_ensemble_types()
    
    output_string_format = "{0: <15}\t{1: <28}\t{8: <15}{2: <13}\t{3: <18}\t{4: <28}\t{5: <22}\t{6: <15}\t{7: <22}"

    # print output header
    print(output_string_format
          .format('score','dataset','dataset_size','number_of_classes','model','ensemble','ensemble_size','voting_system', 'feature_count'))

    for dataset in datasets:

        # load mock_classifier
        X, Y = dataset.load()
        # dataset info
        dataset_size = X.shape[0]
        classes_count = np.unique(Y).size

        
        # split mock_classifier into train and examples sets
        #X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
        kfold_labels = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=1993)
        
        # evaluate models
        for model in models:

            # calculate performance score of the model
            score = model_score_kfold(model, X, Y, kfold_labels, dataset.problem_type())
            
            # print results in csv format
            print(output_string_format
                  .format(score, dataset.name(), dataset_size\
                          ,classes_count, model.name()\
                          ,'NONE', 'NONE', 'NONE', X.shape[1]))

            # score enembles based on current model
            for voting_system in voting_systems:
                
                for ensemble_type in ensemble_types:
                    
                    # create ensemble
                    classifiers_in_ensamble = 30
                    ensemble = ensemble_type(voting_system, type(model), classifiers_in_ensamble)
                    
                    # calculate performance score of ensemble
                    score = model_score_kfold(model, X, Y, kfold_labels, dataset.problem_type())
            
                    # print results in csv format
                    print(output_string_format
                          .format(score, dataset.name(), dataset_size\
                                  ,classes_count, model.name()\
                                  ,ensemble.name(), classifiers_in_ensamble\
                                  ,voting_system.name(), X.shape[1]))
