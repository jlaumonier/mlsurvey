# Backlog

# Done definition
1. Code 
2. Unit test
3. Documentation
4. Pushed in git

## TODO
* Visualization : interface query to nosql database 
* Fairness : rethinking the workflow to integrate supervised learning workflow
* Supervised Learning : use compact config instead of full config
* Allowing tuple in config hyperparameters (e.g. hidden_layer_sizes for MLPClassifier)
* Generate a "full" config file for sci-kit-learn
* Log management
* Rename Logging class to "FilesOperation"-ish class
* Regularization
* Model evaluation (complete)
* Random seed global
* Configuration : add config about the program behavior (log dir, ...)
* Version management : https://github.com/aebrahim/python-git-version ?
* Document tests inside code
* Bug : Remove warning during importlib in Utils
* Invert parameters of assertions in tests
* use a design pattern for persistence of mlsurvey.models
* Reorganize workflow inheritance and attribute (e.g. config)
* Doc website
* Create generic factory
* Fairness utils : make the calculation of conditional proba more robust (more test)l
* Defining and validating a json schema for config files

## Current
* Visualization : interface display one result
* Visualization : interface display multiple results

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




