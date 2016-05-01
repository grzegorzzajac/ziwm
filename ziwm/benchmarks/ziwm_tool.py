#!/usr/bin/python2.7

import numpy as np
import sys
import argparse

from ziwm.model.base_classifier.classifier import Classifier
from ziwm.benchmarks.validator import model_score_kfold
from ziwm.model.voting_system.voting_system import VotingSystem
from ziwm.model.ensemble.ensemble import Ensemble
from ziwm.data.dataset_loader.dataset_loader import DatasetLoader
from sklearn.cross_validation import StratifiedKFold

def __load_datasets():
    dataset_loader = DatasetLoader()
    dataset_loader.load()
    datasets = dataset_loader.get_all_datesets()
    return datasets

def __kfold_labels(Y):
    return StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=1993)

def benchmark_classifiers():

    # load all possible models and datasets
    datasets = __load_datasets()
    models = Classifier.all_models()

    # output format
    output_string_format = "{0},{1},{2},{6},{3},{4},{5}"
    print(output_string_format
         .format('time','score','dataset','dataset_size','number_of_classes','model','feature_count'))

    for dataset in datasets:
        
        # load mock_classifier
        X, Y = dataset.load()
        # dataset info
        dataset_size = X.shape[0]
        classes_count = np.unique(Y).size

        
        # Split dataset into kfold datasets
        kfold_labels = __kfold_labels(Y)
        
        # evaluate models
        for model in models:

            score, total_time = model_score_kfold(model, X, Y, kfold_labels, dataset.problem_type(), measure_time=True)
        
            # print results in csv format
            print(output_string_format
                  .format(total_time, score, dataset.name(), dataset_size\
                          ,classes_count, model.name(), X.shape[1]))
            sys.stdout.flush()
    
def benchmark_ensembles(print_to_file=False):

    # load all possible models and datasets
    datasets = __load_datasets()
    models = Classifier.all_models()
    voting_systems = VotingSystem.all_voting_systems()
    ensemble_types = Ensemble.all_ensemble_types()
    
    # output format
    if print_to_file == True:
        output_string_format = "{0},{1},{8},{2},{3},{4},{5},{6},{7}"
    else:
        output_string_format = "{0: <15}\t{1: <28}\t{8: <15}{2: <13}\t{3: <18}\t{4: <28}\t{5: <22}\t{6: <15}\t{7: <22}"

    # print output header
    print(output_string_format
      .format('score','dataset','dataset_size','number_of_classes','model','ensemble','ensemble_size','voting_system', 'feature_count'))
    
    
    ensamble_sizes = [1, 2, 4, 6, 8, 12, 15, 20, 25, 30, 40, 50, 60, 70, 100]

    for classifiers_in_ensamble in ensamble_sizes:

        for dataset in datasets:
            
            # load mock_classifier
            X, Y = dataset.load()
            # dataset info
            dataset_size = X.shape[0]
            classes_count = np.unique(Y).size
            
            # Split dataset into kfold datasets
            kfold_labels = __kfold_labels(Y)
            
            # evaluate models
            for model in models:
                    
                # score enembles based on current model
                for voting_system in voting_systems:
                    
                    for ensemble_type in ensemble_types:
                        
                        # create ensemble
                        ensemble = ensemble_type(voting_system, type(model), classifiers_in_ensamble)
                        
                        # calculate performance score of ensemble
                        score = model_score_kfold(ensemble, X, Y, kfold_labels, dataset.problem_type())
                
                        # print results in csv format
                        print(output_string_format
                              .format(score, dataset.name(), dataset_size\
                                      ,classes_count, model.name()\
                                      ,ensemble.name(), classifiers_in_ensamble\
                                      ,voting_system.name(), X.shape[1]))
                        sys.stdout.flush()


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark prediction models")
    parser.add_argument("mode", help="Type of models to benchmark",
                        choices=['ensembles', 'classifiers'])
    parser.add_argument("-f", "--file", help="Print results to a file")
    args = parser.parse_args()
    print args.mode
    print_to_file = False
    if args.file != None:
        print "Printing output to '" + args.file + "'file"
        sys.stdout = open(args.file, 'w')
        print_to_file = True
        
    # Run
        
    if args.mode == "ensembles":
        benchmark_ensembles(print_to_file)
    elif args.mode == "classifiers":
        benchmark_classifiers()

if __name__ == '__main__':
    main()