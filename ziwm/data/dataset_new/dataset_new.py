#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.base import Dataset


class DatasetNew(Dataset):
    '''

    '''
    def __init__(self, dir_path, file_name):
        self.__name = file_name[:-5]
        self.__full_path = path.join(dir_path, file_name)

    def name(self):
        return self.__name

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read(self.__full_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

