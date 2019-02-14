# Concepts

1. Input : for classification : x, y
1. Algorithm : 
    1. Operation : Init, set Hyperparameters, Learn
    1. Defintion
        1. Predictor family
        1. Loss function
        1. Optimization method
1. Hyperparameters
    1. Hyperparameters for knn : 
        * n_neighbors 
        * algorithm 
        * weights
1. Result : results of the algorithm
1. Visualization with matplotlib

# Datasets

Sone datasets are defined : Iris (REF), N-class Random data (REF) based on scikit learn.
DataSetFactory can be used to create a DataSet from the name of the dataset
A DataSet can be affected to an input to feed the algorithm
Some parameters are defined for some datasets. Not all parameters defined in sckitlearn can be used

## Existing datasets

1. Circles : make circles (from sklearn.datasets)
1. Iris : Iris dataset (from sklearn.datasets)
1. Moons : make moons (from sklearn.datasets)
1. N Class Random : Make a n class random problem (from sklearn.datasets)

# Exporting results

Results of a learning (input. classifiers, evalution, config) can be stored as json files into a directory Logs/(datehour)/
The results can be loaded be the supervised learning workflow and can be show by Visualization

# Configuration

The configuration of the learning process is defined in the config/config.json and can be loaded and accessed 
from the mlsurvey.Config class. Each element in the learning_process section can be a list to launch multiple learning
However, I am not sure if i will keep DotMap() access. 

# Learning process

The supervised learning process follow this structure :
1. choose the dataset (generate, load, define)
1. input pretreatement (support StandartScaler (mean 0, stddev 1))
1. apply the model evaluation strategy to split the input (support Train+Test)
1. learn
1. model evaluation and metrics calculation (confusion matrix, TODO)

The multiple learning workflow run multiple supervised learning workflow according to the config file

