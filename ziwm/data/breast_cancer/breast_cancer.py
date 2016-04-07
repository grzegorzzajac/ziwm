#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.base import Dataset


class BreastCancer(Dataset):
    '''
       1. Class: no-recurrence-events, recurrence-events
            0, 1
       2. age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
            0, 1, 2, 3, 4 , 5, 6, 7, 8
       3. menopause: lt40, ge40, premeno.
            0, 1, 2
       4. tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
            0, 1, 2, 3, 4 , 5, 6, 7, 8, 9, 10, 11
       5. inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
            0, 1, 2, 3, 4 , 5, 6, 7, 8, 9, 10, 11, 12
       6. node-caps: no, yes.
            0, 1
       7. deg-malig: 1, 2, 3.
       8. breast: left, right.
            0, 1
       9. breast-quad: left-up, left-low, right-up,	right-low, central.
            0, 1, 2, 3, 4
      10. irradiat: no, yes.
            0, 1
    '''

    def name(self):
        return "Breast Cancer dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data_path = path.join(path.dirname(path.abspath(__file__)), "breast_cancer.data")
        data = elm.read(data_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

