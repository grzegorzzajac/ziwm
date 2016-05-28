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
            x_individual = x_test[:, self.feature_subspaces[i]]
            result = self.base_classifiers[i].predict(x_individual)
            results.append(result)
        return self.voting_system.vote(results)

    def train(self, x, y, class_number=-1, feature_labels = []):
        self.feature_subspaces = []

        for i in range(len(self.base_classifiers)):
            new_x, feature_subspace = self.__create_individual_dataset(x, feature_labels)
            self.base_classifiers[i].train(new_x, y, class_number)
            self.feature_subspaces.append(feature_subspace)

    @staticmethod
    def __create_individual_dataset(x, feature_labels):
        feature_labels = np.asarray(feature_labels)
        features_size = np.unique(feature_labels).size
        r = features_size / 2  # number of randomly picked features for subspace
        feature_subspace = random.sample(xrange(features_size), r)

        selected_indices = np.zeros(len(feature_labels)).astype(bool)
        for feature in feature_subspace:
            selected_feature = feature_labels == feature
            selected_indices = np.logical_or(selected_indices, selected_feature)

        x_selected = x[:, selected_indices]
        return x_selected, selected_indices
