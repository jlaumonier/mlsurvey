# Backlog

# Done definition
1. Code 
2. Unit test
3. Documentation
4. Pushed in git

## TODO
* Visualization : Improve process : use bokeh server and dynamic interface. Refactor workflow
* Log management
* Regularization
* Model evaluation (complete)
* Random seed global
* Error Management - Make the system robust
* Configuration : add config about the program behavior (log dir, ...)
* Version management : https://github.com/aebrahim/python-git-version ?
* Document tests inside code
* Remove warning during importlib in Utils
* Invert parameters of assertions in tests
* use a design pattern for persistence of mlsurvey.models

## Current
* split supervised_learning_workflow to separate data from workflow. Data should be using in visualization_workflow
* Reorganize Dataset and Data : concept are too similar and not well defined

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



