#!/usr/bin/python2.7

import numpy as np
import argparse
from multiprocessing.pool import Pool
from multiprocessing.synchronize import Lock
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import sys

from ziwm.model.classifier.base import Classifier
from ziwm.benchmarks.validator import model_score_kfold
from ziwm.utils.voting_system.voting_system import VotingSystem
from ziwm.model.ensemble.ensemble import Ensemble
from ziwm.data.dataset_loader.dataset_loader import DatasetLoader
from sklearn.cross_validation import StratifiedKFold
from ziwm.model.ensemble.random_subspace.random_subspace import RandomSubspace

def __load_datasets():
    dataset_loader = DatasetLoader()
    dataset_loader.load()
    datasets = dataset_loader.get_all_datesets()
    return datasets

def __kfold_labels(Y):
    return StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=1993)

def __load_feature_labels(filename):
    f = open(filename, 'r')
    header = f.readline().rstrip()
    str_labels = header.split(',')
    str_labels = list(filter(lambda s : s.isdigit() or s == '-1', str_labels))
    labels = list(map(int, str_labels))
    return labels

def __model_score_job(args):
    ensemble, X, Y, kfold_labels, dataset, model, output_string_format, classifiers_in_ensamble, voting_system_name, feature_labels = args
    # dataset info
    dataset_size = X.shape[0]
    classes_count = np.unique(Y).size

    score = None
    try:
        score = model_score_kfold(ensemble, X, Y, kfold_labels, dataset.problem_type(), feature_labels=feature_labels)
    except AssertionError:
        lock.acquire()
        print("Assert:" + str(ensemble) + "," + dataset.name())
        sys.stdout.flush()
        lock.release()
        return -1.0

    lock.acquire()
    print(output_string_format
                .format(score, dataset.name(), dataset_size\
                        ,classes_count, model.name()\
                        ,ensemble.name(), classifiers_in_ensamble\
                        ,voting_system_name, X.shape[1]))
    sys.stdout.flush()

    lock.release()
    return score

def __init_proc(l):
    global lock
    lock = l

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
    
    
    ensamble_sizes = [1, 2, 4, 6, 8, 12, 15, 20, 25, 30, 40, 50, 60, 70, 100][::-1]

    process_jobs_args = []
    l = Lock()

    for classifiers_in_ensamble in ensamble_sizes:

        for dataset in datasets:
            
            # load mock_classifier
            X, Y = dataset.load()
            # dataset info
            dataset_size = X.shape[0]
            classes_count = np.unique(Y).size
            
            # Split dataset into kfold datasets
            kfold_labels = __kfold_labels(Y)

            # Check if every class is in every set after kfold
            for train_index, test_index in kfold_labels:
                Y_train, Y_test = Y[train_index], Y[test_index]
                assert np.unique(Y_train).size == classes_count
            
            # evaluate models
            for model in models:
                    
                # score enembles based on current model
                for voting_system in voting_systems:
                    
                    for ensemble_type in ensemble_types:

                        feature_labels = None
                        if ensemble_type == RandomSubspace:
                            feature_labels = __load_feature_labels(dataset.path())

                        # create ensemble
                        ensemble = ensemble_type(voting_system, type(model), classifiers_in_ensamble)
                        
                        job_args = ensemble, X, Y, kfold_labels, dataset, model, output_string_format, classifiers_in_ensamble, voting_system.name(), feature_labels
                        process_jobs_args.append(job_args)
                        
    pool = Pool(initializer=__init_proc, initargs=(l, ), processes=None)
    pool.map(__model_score_job, process_jobs_args)
    pool.terminate()
    pool = None


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark prediction models")
    parser.add_argument("mode", help="Type of models to benchmark",
                        choices=['ensembles', 'classifiers'])
    parser.add_argument("-f", "--file", help="Print results to a file")
    args = parser.parse_args()
    print "Benchmarking", args.mode
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
