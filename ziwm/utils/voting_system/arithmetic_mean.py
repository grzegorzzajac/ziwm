#!/usr/bin/python2.7

import numpy as np
from ziwm.utils.voting_system.voting_system import VotingSystem


class ArithmeticMean(VotingSystem):
    '''
    Voting System returning arithmetic mean of given results
    '''

    @staticmethod
    def name():
        return "Arithmetic Mean"

    @staticmethod
    def vote(results):
        avg = np.average(results, axis=0)
        # rnd = np.round(avg)
        return avg

