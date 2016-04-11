#!/usr/bin/python2.7

from ziwm.data.dataset_loader.dataset_loader import DatasetLoader

dsl = DatasetLoader()
dsl.load()

for ds in dsl.get_all_datesets():
    print '<< ', ds.name(), ' >>'
    x, y = ds.load()
    print x, y, '\n'

ds = dsl.get_next_dataset()
while ds is not None:
    print '<< ', ds.name(), ' >>'
    x, y = ds.load()
    print x, y, '\n'
    ds = dsl.get_next_dataset()

