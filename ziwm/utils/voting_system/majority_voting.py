#!/usr/bin/python2.7

import numpy as np
from ziwm.utils.voting_system.voting_system import VotingSystem

class MajorityVoting(VotingSystem):
    '''
    Voting System counting votes for particular classes and returning the most frequent ones as a final result
    '''

    @staticmethod
    def name():
        return "Majority Voting"

    @staticmethod
    def vote(results):
        class_num = np.asarray(results).shape[2]
        results_transposed = map(list, zip(*results))
        #print 'rezulataty transponowane: ', results_transposed
        voting_result = []
        for row in results_transposed:
            rounded_row = np.argmax(row, axis=1)
            counts = np.bincount(rounded_row, minlength=class_num)
            voting_result.append(counts)
        #print 'wyniki glosowania: ', voting_result
        return np.asarray(voting_result)

