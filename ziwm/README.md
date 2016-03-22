# ziwm

## ziwm_tool

Module aims to evaluate accuracy and speed of various Machine Learning algorithms over different datasets.

It trains each available algorithm and evaluates it's performance on each available dataset. Results of such tests are printed to stdout in CSV format, e.g:
```
dataset,model,score
mock_dataset,MockClassifier,1.0
```


*ziwm_tool* runs tests on all models registered in *all_models()* method in _Model_ class and all datasets registered in *all_datasets()* method in _Dataset_ class. **If new model or dataset is added it should be registred in those places.**

## Datasets

Every available dataset is represented by a separate class extended from _Dataset_ class. It is responsible for reading data from file and creating features from data. An example of such a class is *MockDataset* which simply returns some arbitrary data.

# Models

Every prediction model has it's own class extended from _Model_ class. An example of such a model is MockClassifier class.
