#!/usr/bin/python2.7

import numpy as np
from ziwm.model.voting_system.voting_system import VotingSystem


class ArithmeticMean(VotingSystem):
    '''
    Voting System returning arithmetic mean of given results
    '''

    @staticmethod
    def name():
        return "Arithmetic Mean"

    @staticmethod
    def vote(results):
        return np.average(results, axis=0)

