#!/usr/bin/python2.7

import random
import numpy as np
from ziwm.model.ensemble.ensemble import Ensemble


class RandomSubspace(Ensemble):

    #def __init__(self, base_classifiers, voting_system):
    #    super(RandomSubspace, self).__init__(base_classifiers, voting_system)
    #    self.feature_subspaces = []
    
    def __init__(self, voting_system, classifier_type, classifier_count):
        super(RandomSubspace, self).__init__(voting_system, classifier_type, classifier_count)

    @staticmethod
    def name():
        return "Random Subspace"

    def predict(self, x_test):
        results = []
        assert len(self.feature_subspaces) == len(self.base_classifiers)
        for i in range(len(self.base_classifiers)):
            x_individual = self.__create_x_from_feature_subspace(x_test, self.feature_subspaces[i])
            result = self.base_classifiers[i].predict(x_individual)
            results.append(result)
        return self.voting_system.vote(results)

    def train(self, x, y, class_number=-1):
        self.feature_subspaces = []
        for i in range(len(self.base_classifiers)):
            new_x, feature_subspace = self.__create_individual_dataset(x)
            self.base_classifiers[i].train(new_x, y, class_number)
            self.feature_subspaces.append(feature_subspace)

    @staticmethod
    def __create_individual_dataset(x):
        features_size = x.shape[1]
        r = features_size / 2  # number of randomly picked features for subspace
        feature_subspace = random.sample(xrange(features_size), r)
        new_x = np.zeros((x.shape[0], r))
        for i in range(r):
            feature = feature_subspace[i]
            new_x[:, i] = x[:, feature]
        return new_x, feature_subspace

    @staticmethod
    def __create_x_from_feature_subspace(x, feature_subspace):
        new_x = np.zeros((x.shape[0], len(feature_subspace)))
        for i in range(len(feature_subspace)):
            new_x[:, i] = x[:, feature_subspace[i]]
        return new_x
