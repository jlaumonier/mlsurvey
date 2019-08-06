# Backlog

## Done definition
1. Code 
2. Unit test
3. Documentation
4. Pushed in git

## Notes for TODO
* B is Bug
* F is functionality
* T is technical debt
* A is Architecture

## TODO
* F : Change config format to introduction definition section for non compact format
* F : Use panda dataframe into Data() and Dataset() (big changes...)
* F : Generate a "full" config file for sci-kit-learn
* F : Log management
* B : Visualization workflow does not handle correctly more than 2 classes
* T : Rename Logging class to "FilesOperation"-ish class. Make it used by the Config class to read the json file
* F : Regularization
* F : Model evaluation (complete)
* F : Random seed global
* F : Configuration : add config about the program behavior (log dir, ...)
* T : Version management : https://github.com/aebrahim/python-git-version ?
* T : Document tests inside code
* B : Remove warning during importlib in Utils
* T : use a design pattern for persistence of mlsurvey.models
* T : Reorganize workflow inheritance and attribute (e.g. config)
* T : Doc website
* F : Create generic factory
* T : Fairness utils : make the calculation of conditional proba more robust (more test)
* F : Defining and validating a json schema for config files
* B : Scikit learning : use a predict() on classifier change the joblib saved file (before and after predict()). Write small test program  
* A : Analyse if y_pred is in a correct location (Data)
* F : dash adding dcc.Loading. Warning, this provokes bug in display (update option, blank page)
* B : joblib md5 depends on the version of python/joblib... :S search other method to test the save file. Not simple
        See https://datascience.stackexchange.com/questions/33527/scikit-learn-decision-tree-in-production
        https://github.com/scikit-learn/scikit-learn/issues/11041
        https://www.andrey-melentyev.com/model-interoperability.html
* F : adding search of specific dataset. Adding a new search field ?
* F : adding a base_directory for LearningWorkflows and child
* B : Demographic parity is not defined if, in a dataset, no instance belong to privileged or unpriviliged class.

## Current
* F : Fairness : rethinking the workflow to integrate supervised learning workflow

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
* Adding display options to interface
* Bug : Dash display blank page when all results are deselected : remove Loading component
* A : Regenerate/rethinking the test files (md5 or other database ?)



