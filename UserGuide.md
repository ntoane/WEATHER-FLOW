# User Guide to the Experimental Platform

*This is an expanded version of the initial ReadMe.md file, that elaborates on the usage, customizability of the experimental platform and provides a step by step process of integrating a new cutting-edge ST-GNN model implementation.*

## Introduction

* Overview of the experimental platform

  * This experimental platform was initially created by Davidson for [his Masters project at UCT](https://www.springerprofessional.de/en/st-gnns-for-weather-prediction-in-south-africa/23774860). It was initially not designed for future use, but since then has become enhanced to be a clear, easy-to-use experimental platform for the Adaptive and Cognitive Systems Lab, that's customizable and extensible with code modularity maximized and code coupling minimized as much as possible.
* Aims & Objectives of the experimental platform

  * Extend Davidson’s experimental platform to evaluate all our individual ST-GNN variants, along with the baseline models.
  * Ensure that the adjacency matrix depicting the weather stations’ dependencies on each other is effectively captured by the visualisation methods.
  * Validate and evaluate the developed experimental platform using the baseline techniques and our individual techniques.
  * Ensure that the experimental platform is highly usable, easily integratable for new techniques and well designed to allow for easy future use.
* 

## Installation & Setup

* System requirements:

  * Hardware:

    - CPU: A multi-core processor (e.g., Intel Core i5 or higher) or a server-grade CPU would be suitable. More cores can speed up computations in parallel.
    - Memory (RAM): At least 8 GB of RAM is recommended, although more may be required for larger datasets or complex models.
    - GPU (optional): If you plan to leverage GPU acceleration for faster training, a dedicated GPU with CUDA support, such as NVIDIA GeForce or Tesla series, would be beneficial.
  * Software:

    * Operating System: The experimental platform can be developed on various operating systems, including Windows, macOS, or Linux.
    * Development Environment: Use an integrated development environment (IDE) like Jupyter Notebook, PyCharm, or Anaconda to facilitate code development, debugging, and experimentation.
    * Python: Both GWN and TCN models can be implemented using Python. Install the latest version of Python and the necessary libraries, such as TensorFlow or PyTorch, depending on the specific framework used.
* Installation instructions for Python and necessary libraries:

  * Same as seen in the Readme.md
  * Please refer to FAQ's for any installation errors, as these may be frequently encountered errors that can be guided through with some tips and tricks.

## New Model Integration

These are the steps to follow when integrating a new ST-GNN model (such as *<model_name>*) implementation to the experimental platform :

* Configurations settings for <model_name> :
  * Create a new config.yaml file for the model in *configurations/"<model_name>Config.yaml"*
  * Use/add any configuration settings that are general to *"sharedConfig.yaml"* file
* Module placing for <model_name> :
  * Model logic goes in *Models/<model_name>/"<model_name>.py"*
  * Training Logic goes in the *Train/"<model_name>Train.py"*
  * Hyper-parameter optimization logic goes in *HPO/"<model_name>HPO.py*
  * Evaluation logic in the *Evaluation/"<model_name>eval.py"*
* Shared data for experimentation is in *DataNew/* module :
  * Depending on methodology of the cutting-edge ST-GNN model, either can obtain an:
    * Adjacency matrix -> "adj_mx.pkl"
      * Found in *DataNew/Graph_Neural_Network_Data/Adjacency_Matrix/adj_mx.pkl*
      * *Please note that .pkl files can be 'unpickled' and converted into a readable fomat (such as a .csv file) by the function load_pickle in Utils/gwnUtils.py*
    * .csv fie with all the weather station data :
      * Either in one large .csv file
        * Located in DataNew/Graph_Neural_Network_data/Graph_Station_Data
      * Or one weather station in a single .csv file
        * Located in DataNew/Weather_Station_Data
* Logs :
  * The "*/Logs/modelLogger"* logic is available to be implemented in the code of the model implementation, following the pre-defined steps in the Logging metrics subsection.
  * Logging information can be saved to *Logs/<model_name>/<run_type</<model_name>*
    * <run_type> is the type of experiment with the model, usually either Training, HPO or Evaluation
* Visualisation/ module:
  * This module has "visualise.py" which contains the methods used for visualising the the adjacency matrices collected from the experimentation.
  * Currently visualised into either an adjacency matrix heatmap, strongest dependencies and strong chains of dependencies on a map.
  * Saved to *Visualisations/<model_name>/<horizon>/visualisation_type>/*
* Plots/ module:
  * Module Plots/ has "plotter.py" which contains the logic for plotting graphs with the sepcified metrics
  * Saved to *Plots/<module_name>/`<metric><plot>.jpg`*
  * Currently plots box and whiskers diagram

## Configurations

This subsection describes the methodology of the configurations/ module that sets up the experimental platform with the specified model implementation and desired settings.

* Shared config settings in *"configurations/sharedConfig.yaml"* that has general data configurations that all models may use:
  * Boolean settings to select different models and types of experiments (training, HPO, or evaluation). Multiple settings can be set to true, enabling sequential training of different models. For example, if a user sets "train_tcn" and "train_gwn" to true, the platform will first train the TCN model and then proceed to train the GWN model. Users can leave their PC running, and both models will complete their training when they return.
  * General data configurations include increment steps, list of stations in data and the numbe rof horizons etc.
  * Common loss function and optimizer can be set in this file, however a model only uses it if its own configuration file specifies that it is not using an independent method (that is it is using the common settings)
  * Visualization settings are also found here, with its relevant settings.
* Model specific config settings in "*configurations/<model_name>Config.yaml*" that's only used by a single model:
  * This Yaml file contains settings that are used in the configuration and initialisation of a specific model.
  * If an independent loss function or optimizer wants to be used set appropriate boolean flag to true and set method desired (eg. set use_optimizer to true and set optimizer to SGD if Stochastic Gradient Descent is required). Default settings for loss function and optimizer is MSE and Adam respectively
* Please note: Current config settings available for following:
  * Loss function can be set to:
    * MSE, MAE, sparse_categorical_crossentropy, categorical_crossentropy.
  * Optimizers can be set to:
    * Adam, SGD, RMSprop.
  * If desired method/settings for above is not in options then it needs to be set in Model and Train code of the model in question.

## Visualisations

* Visualisations on the adjacency matrix can be generated using this platform.
* There are 3 visualisations:
  * Heatmap
  * Strongest Dependencies
  * Influencial Paths
* How to use it
  * To make use of the visualisation, you need to open to the sharedConfig.yaml file and naviagte to the visualisation section.
  * Change the parameter setting for visualisations to true and proceed to running the platform with the standard command "*python3 main.py --mode config.yaml*"
  * Further adjust the parameters to the specific matrix needed to be visualised.
  * Next adjust other parameters to fine tune the visualisations.

## Data Preprocessing

* Description
  * We have currently gathered weather stations for Eastern Cape, Northern Cape and Western Cape provided by South African Weather Services (SAWS), of 45 weather stations in total, 15 stations each per province. This is more than previously used by Davidson.
* Steps for data preprocessing and cleaning
  * Python script was wriiten to ensure data was filtered according to individual stations and only stations that had below 5% missing data were selected for the learning process
* Handling missing values and time records
  * We implemented Inverse Distance Weigthing method (IDW) to handle the missing time records or variables in the data
* Splitting the data into training and test sets
  * Data was split using walk forward validation method

## Experimental pipeline

Brief explanation of the experimental pipeline used for, GWN, and TCN as baselines:

* TCN:
  * Separate ***TCN models are built for each weather station.***
  * Walk-forward validation is performed on each station's dataset, resulting in multiple training, validation, and test splits.
  * Hyper-parameter optimization is done for the first three splits, considering parameters such as epochs, batch size, learning rate, input window size, number of layers, dropout rate, layer normalization, and number of filters.
  * The optimal parameters are selected based on the lowest error across all splits for the 24-hour prediction horizon.
  * The models are trained and evaluated for each split, and the predictions for each split are concatenated.
  * This has been deemed to be a respectable baseline to compare other, newer models, prediction accuracy on.
  * SMAPE metrics are calculated using the concatenated predicted values and the actual values.
* GWN:
  * ***A single model is built for all stations***.
  * GWN is a variant of ST-GNN that combines TCN layers and graph convolutional layers.
  * It utilizes a self-adaptive adjacency matrix learned through gradient descent in the graph convolutional module.
  * The experimental pipeline for these models follows a similar process as the TCN model, but with some differences. In addition to TCN's hyper-parameters, the adjacency matrices need to be considered.
  * The adjacency matrix is of the self-adaptive type and is randomly initialized. This model had evidently become a reputable baseline that other, cutting-edge models, can be effectively compared against.
* *Note for more details on the baselines please refer to our Proposal document.*

## Customization & Extension

Guidelines for customizing the experimental platform in terms of Training, Prediction & Evaluation, Logging, Visualization

* Adding new models or modifying existing ones

  * Follow New Model Integration step by step instructions.
* Training custimizations:

  * This includes the process of setting up the training process of a specific model

    * This varies according to your model structure and needs to be defined within the Execute module in a python file.
  * For setting loss functions, optimizers, and learning rates

    * This can be done by navigating to the config files and altering the relevant option
* Prediction & evaluations:

  * Using the trained models for weather prediction
  * Evaluation metrics such as MSE, RMSE, MAE and SMAPE are already included within the platform and can be easily used for an analysis on your model.
  * The baseline models of TCN and GWN are readily functioning on the repo and can be used as a baseline for comparing your model against.
* Logging module:

  * The **modelLogger** class is essentially a wrapper around Python's built-in logging  library. It provides a way to easily log different levels of information (i.e., info, debug, warning, error and critical) from a specific model running on a specific station to a log file. It also allows enabling or disabling logging through the log_enabled parameter.
    * In Logs module, under each model,  the logs will be stored for the various stages of experimentation.
    * For example for training, the logs will be in the Train folder under the relevant horizons.
    * Actuals vs Predicted results are captured here - alongside the time indices of when they occurred in the data for analysis purposes. As if one of the models has seemingly unexplainable bad accuracy for some predictions, we can check against the logs and see if it was all around the same time and we can have a better idea of what happened.
    * Other admin logging content is also captured here at the various logging levels described in the modelLogger class.
  * The **Evaluation class**
    * This is a Python class that holds multiple methods that pertain to the evaluation process of each specific model. This includes the calculations of the specified metrics with the actual and predicted values obtained (normalized ones) and the writing and logging to files thereof.

## Troubleshooting & FAQs

* Common issues and solutions:
  * Dependency or library conflicts:
    * Ensure you are using a virtual enviornment with modules in the requirements_all.txt correctly installed.
    * Check on [tensorflow](https://www.tensorflow.org/guide/versions) or [pytorch](https://pytorch.org/docs/stable/index.html) for a specific version that's compatible, even if it is a lower version than the one currently stated in the requirements.txt file.
    * Downgrade specific modules such as scki-learn or pandas, to make sure they don't intefere with anything.
    * Also if it is only a specific model not working, ie a more complex one such as GWN, then refer to the original code base for that and try get that working isolation. Then afterwards you will have a better idea on the packages and versions needed to get it running, and you can try adapt them to fit in the experimental platform.
    * Can always manually install the module with your computer's package manager which can work. Don't forget to set the path to the installed package.
    * If *Basemap* package throws errors during installation you can comment it and install it after all the packages by running *pip install basemap==1.3.7 --user*
* Frequently asked questions about the platform and its usage:
  * Can the platform handle large-scale datasets?

    * Yes currently the platform is able to handle such large datasets, an example is a .csv file with several weather columns of data that spands about 5 126 800 rows.
  * How can I fine-tune or customize the TCN and GWN models within the platform?

    * Yes such customization settings can be located within the config.yaml file under the relevant section.
  * How accurate are the TCN and GWN baseline models in weather temperature prediction?

    * These models have been chosen as baselines as they are evidently amongst the most reputable within the ST-GNN domain. Although they haven't explicitly been used for the weather temperature prediction problem, they will most likely yield impressive results, given their success in the Traffic prediction domain.
  * Are there any specific data requirements or preprocessing steps for using this platform?

    * Currently we have only used data provided by the  South African Weather Services (SAWS), that is in a certain column format. We have created a module dedicated to pre-processing the data.
  * Can I use my own spatial-temporal GNN models with this platform

    * Yes that's the primary goal of this experimental platform, is for it to be highly usable and easily integrated with any cutting-edge ST-GNN technoque that may arise in the following years.

## References & Resources

* List of relevant and foundational research papers, articles, and books:

  * Foundational Theory:

    * [Davidsons Paper](https://www.springerprofessional.de/en/st-gnns-for-weather-prediction-in-south-africa/23774860)
    * [Pillays Paper](https://link.springer.com/chapter/10.1007/978-3-030-95070-5_7)
  * Experimental Platform:

    * [Concept #1 ](https://pure.mpg.de/rest/items/item_3020343/component/file_3036194/content)
    * [Concept #2](https://openaccess.thecvf.com/content_ICCV_2019/html/Savva_Habitat_A_Platform_for_Embodied_AI_Research_ICCV_2019_paper.html)
    * [Concept #3](https://www.emerald.com/insight/content/doi/10.1108/JMTM-02-2022-0092/full/html)
  * Visualizations

    * [Concept #1](https://link.springer.com/chapter/10.1007/978-3-031-22321-1_7)
    * [Concept #2](https://www.sciencedirect.com/science/article/pii/S0950705122000508?casa_token=3CtDCSkfVKMAAAAA:mjnnTlS7FpjsFqTF7xEY0nj7CH5fjse7kVukBX5AWVVYemiWDWxMK31MQXuiTk4o_5P_a6-5Zew)
* Links to external resources and libraries used in the platform that are most likely to cause version and dependency errors:

  * [Tensorflow](https://www.tensorflow.org)
  * [Pytorch](https://pytorch.org)
  * [NumPy](https://numpy.org/doc/stable/)
  * [Pandas](https://pandas.pydata.org/docs/user_guide/index.html)
  * [Yaml Configuration Files](https://elib.psu.by/handle/123456789/36942)
