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
        results_transposed = map(list, zip(*results))
        #print 'rezulataty transponowane: ', results_transposed
        voting_result = []
        for row in results_transposed:
            rounded_row = [abs(int(round(n, 0))) for n in row]
            counts = np.bincount(rounded_row)
            voting_result.append(np.argmax(counts))
        #print 'wyniki glosowania: ', voting_result
        return np.asarray(voting_result)

