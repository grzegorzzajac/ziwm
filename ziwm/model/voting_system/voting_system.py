from abc import ABCMeta, abstractmethod


class VotingSystem(object):
    '''
    Base class for voting systems
    '''
    __metaclass__ = ABCMeta

    @staticmethod
    def all_voting_systems():
        '''
        Returns objects of all available voting systems classes
        (should be subclasses of VotingSystem class)
        '''
        voting_systems = []
        return voting_systems

    @staticmethod
    @abstractmethod
    def name():
        '''
        Arbitrary name of a given voting system
        '''
        pass

    @staticmethod
    @abstractmethod
    def vote(all_results):
        '''
        Make final decision basing on multiple results given
        '''
        pass

