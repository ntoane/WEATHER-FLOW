# User Guide to the Experimental Platform

*This is an expanded version of the initial ReadMe.md file, that elaborates on the usage, customizability of the experimental platform when 'plugging in' future cutting-edge ST-GNN models.*

## Introduction

* Overview of the experimental platform

  * This experimental platform was initially created by Davidson for [his Masters project at UCT](https://www.springerprofessional.de/en/st-gnns-for-weather-prediction-in-south-africa/23774860). It was initially not designed for future use, but since then has become enhanced to be a clear, easy-to-use experimental platform for the cognitive systems lab, that's customizable and extensible with code modularity maximized and code coupling minimized as much as possible.
* Aims & Objectives of the experimental platform

  * Extend Davidson’s experimental platform to evaluate all our individual ST-GNN variants, along with the baseline models.
  * Ensure that the adjacency matrix depicting the weather stations’ dependencies on each other is effectively captured by the visualisation methods.
  * Validate and evaluate the developed experimental platform using the baseline techniques and our individual techniques.
  * Ensure that the experimental platform is highly usable, easily integratable for new techniques and well designed to allow for easy future use.
* Brief explanation of the experimental pipeline used for, GWN, and TCN as baselines:

  * TCN:
    * Separate ***TCN models are built for each weather station.***
    * Walk-forward validation is performed on each station's dataset, resulting in multiple training, validation, and test splits.
    * Hyper-parameter optimization is done for the first three splits, considering parameters such as epochs, batch size, learning rate, input window size, number of layers, dropout rate, layer normalization, and number of filters.
    * The optimal parameters are selected based on the lowest error across all splits for the 24-hour prediction horizon.
    * The models are trained and evaluated for each split, and the predictions for each split are concatenated.
    * This has been deemed to be a respectable baseline to compare other, newer models, prediction accuracy on.
    * SMAPE metrics are calculated using the concatenated predicted values and the actual values.
  * GWN:
    * GWN is a variant of ST-GNN that combines TCN layers and graph convolutional layers.
    * ***A single model is built for all stations***.
    * It utilizes a self-adaptive adjacency matrix learned through gradient descent in the graph convolutional module.
    * The experimental pipeline for these models follows a similar process as the TCN model, but with some differences. In addition to TCN's hyper-parameters, the adjacency matrices need to be considered.
    * The adjacency matrix is of the self-adaptive type and is randomly initialized. This model had evidently become a reputable baseline that other, cutting-edge models, can be effectively compared against.
  * *Note for more details on the baselines please refer to our Proposal document.*

## Installation & Setup & Configuration

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
* Configuration settings for setting up the experimental platform with the models

  * *config.yaml* file contains all the configuration settings, whereby for the boolean setting the default values can be adjusted to be true or false depending on the model and experiment intended to be run. The other settings that have the default initial values such as the hidden layers etc. the default numbers can be altered too for future experiments.

## Data Preprocessing

* Description
  * We have currently gathered weather stations for Eastern Cape, Northern Cape and Western Cape provided by South African Weather Services (SAWS), totalling at 45 weather stations in total, 15 stations each per province. This is more than previously used by Davidson.
* Steps for data preprocessing and cleaning
  * Python script was wriiten to ensure data was filtered according to individual stations and only stations that had below 5% missing data were selected
* Handling missing values and time records
  * We implemented Inverse Distance Weigthing method (IDW) to handle the missing time records or variables in the data
* Splitting the data into training and test sets
  * Data was split using walk forward validtion method
  
## Model Architecture

* Overview of the neural network models used (GWN and TCN)
* Detailed explanation of the architecture and components of each model
* Configuration options for customizing the models (e.g., number of layers, hidden units, etc.)

## Training

* Setting up the training process
* Defining loss functions, optimizers, and learning rates
* Training procedures and hyperparameter tuning
* Monitoring and logging training progress

## Prediction & Evaluation

* Using the trained models for weather prediction
* Instructions for making predictions on new data
* Evaluation metrics and performance analysis (MSE, RMSE, MAE, etc.)
* Comparing the results of GWN and TCN baselines

## Logging metrics & Visualization

## Customization & Extension

* Guidelines for customizing the experimental platform
* Adding new models or modifying existing ones
* Incorporating additional data sources or features

## Troubleshooting & FAQs

* Common issues and solutions:
  * Dependency or library conflicts:
    * Ensure you are using a virtual enviornment with modules in the requirements_all.txt correctly installed.
    * Check on [tensorflow](https://www.tensorflow.org/guide/versions) or [pytorch](https://pytorch.org/docs/stable/index.html) for a specific version that's compatible, even if it is a lower version than the one currently stated in the requirements.txt file.
    * Downgrade specific modules such as scki-learn or pandas, to make sure they don't intefere with anything.
    * Also if it is only a specific model not working, ie a more complex one such as GWN, then refer to the original code base for that and try get that working isolation. Then afterwards you will after have a better idea on the packages and versions needed to get it running, and you can try adapt them to fit in the experimental platform.
    * Can always manually install the module with your computer's package manager which can work. Don't forget to set the path to the installed package.
* Frequently asked questions about the platform and its usage:
  * Can the platform handle large-scale datasets?
  * How can I fine-tune or customize the TCN and GWN models within the platform?
  * How accurate are the TCN and GWN models in weather temperature prediction?
  * Are there any specific data requirements or preprocessing steps for using this platform?
  * What are the advantages of using TCN and GWN models as baselines?
  * Can I use my own spatial-temporal GNN models with this platform

## References & Resources

* List of relevant and foundational research papers, articles, and books:
  * [Davidsons Paper](https://www.springerprofessional.de/en/st-gnns-for-weather-prediction-in-south-africa/23774860)
  * [Yaml Configuration Files](https://elib.psu.by/handle/123456789/36942)
  * [Experimental Platform Concept #1 ](https://pure.mpg.de/rest/items/item_3020343/component/file_3036194/content),  [Experimental Platform Concept #2](https://openaccess.thecvf.com/content_ICCV_2019/html/Savva_Habitat_A_Platform_for_Embodied_AI_Research_ICCV_2019_paper.html)
  * 
* Links to external resources and libraries used in the platform that are most likely to cause version and dependency errors:
  * [Tensorflow](https://www.tensorflow.org)
  * [Pytorch](https://pytorch.org)
  * [NumPy](https://numpy.org/doc/stable/)
  * [Pandas](https://pandas.pydata.org/docs/user_guide/index.html)
  * 
* 

## Appendix

* Sample code snippets and examples
* Data format specifications
* Additional technical details or supplementary information
