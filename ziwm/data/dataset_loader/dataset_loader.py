#!/usr/bin/python2.7

from os import path
import os
from ziwm.data.dataset_new.dataset_new import DatasetNew


class DatasetLoader(object):
    '''
    Dataset loader scans default (or given) directory searching *.data files and returns file names of datasets.
    '''

    def __init__(self, dir_path=path.join(path.dirname(path.abspath(__file__)), "..", "datasets", "classification")):
        self.__datasets = []
        self.__iterator = 0
        self.__loaded = False
        self.__path = dir_path

    def load(self):
        for f in os.listdir(self.__path):
            if f.endswith(".data"):
                ds = DatasetNew(self.__path, f)
                self.__datasets.append(ds)
        return

    def get_all_datesets(self):
        return self.__datasets

    def get_next_dataset(self):
        #return self.__datasets[self.__iterator % len(self.__datasets)]
        if self.__iterator < len(self.__datasets):
            next_dataset = self.__datasets[self.__iterator]
            self.__iterator += 1
            return next_dataset
        return None

    def load_next_dataset(self):
        return
