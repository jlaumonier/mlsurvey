# Backlog

# Done definition
1. Code 
2. Unit test
3. Documentation
4. Pushed in git

## TODO
* Regenerate/rethinking the test files (md5 or other database ?)
* Bug : joblib md5 depends on the version of python/joblib... :S search other method to test the save file
* Change config format to introduction definition section for non compact format
* Use panda dataframe into Data() and Dataset() (big changes...)
* Fairness : rethinking the workflow to integrate supervised learning workflow
* Generate a "full" config file for sci-kit-learn
* Log management
* Bug : Visualization workflow does not handle correctly more than 2 classes
* Rename Logging class to "FilesOperation"-ish class. Make it used by the Config class to read the json file
* Regularization
* Model evaluation (complete)
* Random seed global
* Configuration : add config about the program behavior (log dir, ...)
* Version management : https://github.com/aebrahim/python-git-version ?
* Document tests inside code
* Bug : Remove warning during importlib in Utils
* use a design pattern for persistence of mlsurvey.models
* Reorganize workflow inheritance and attribute (e.g. config)
* Doc website
* Create generic factory
* Fairness utils : make the calculation of conditional proba more robust (more test)
* Defining and validating a json schema for config files
* Bug Scikit learning : use a predict() on classifier change the joblib saved file (before and after predict()). Write small test program  
* Analyse if y_pred is in a correct location (Data)
* dash adding dcc.Loading. Warning, this provokes bug in display (update option, blank page)

## Current
* Adding display options to interface
* Bug : Dash display blank page when all results are deselected : remove Loading component

## Done
* Git repository : local
* Basic architecture
* Make a mlsurvey package including all classes
* BUG : datasetfactory pas sur d'ajouter des factory automatiquement
* Data data : Moons, circles, linearly separable
* Git repository : gitlab
* Result management : input data
* Visualizing input from file
* Parameters for datasets
* Visualizing classifier results
* Configuration management
* Supervised learning process
* save and load classifier and results
* workflow for multiple datasets, algo and split
* Document supervisedlearningworkflow.py
* Improve multiple learning
* Do not keep DotMap in Config 
* Implements other algorithms and datasets - Generalization
* Multiple-learning workflow : use concurent learning processes
* Complete Visualization : load directory and display data, score and config
* Bug : package management. submodule dataset not correctly handled (mls.XX instead of mls.datasets.XX)
* Visualization of probabilities
* split supervised_learning_workflow to separate data from workflow. Data should be using in visualization_workflow
* Reorganize Dataset and Data : concept are too similar and not well defined
* Fairness workflow
* Visualization : Improve process : use dash  and dynamic interface. Refactor workflow
* Bug : empty log directory during visualization
* Error Management - Make the system robust
* progress bar for multiple learning workflow
* Visualization : all config into nosql database
* Visualization : interface display one result
* Visualization : interface display multiple results
* Bug : Visualization workflow crash for more than 2 classes with algorithm with decision_boundary
* Invert parameters of assertions in tests assertEqual (expected, actual)
* Visualization : basic interface query to nosql database 
* Allowing tuple in config hyperparameters (e.g. hidden_layer_sizes for MLPClassifier)
* save tuple in config
* using german credit dataset
* Bug : Visualization workflow crash when dataset has more than 2 dimensions (e.g. load_iris)
* Adding confusion matrix
* Generate, save and display predicted y for test. 
* Improve the confusion matrix visual (which dimension is true and predicted)
* Adding show/hide sections




