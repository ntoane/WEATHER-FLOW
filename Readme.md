# Weather-Flow
test server github

*The Deep Learning pipeline for weather prediction, developed as an experimental platform for the UCT Cognitive Systems Lab, incorporates temporal and spatio-temporal deep learning models. This pipeline aims to facilitate experimentation and integration of the latest and most cutting-edge ST-GNN (Spatio-Temporal Graph Neural Network) models for weather forecasting.*

| Member              | Student Number |
| ------------------- | -------------- |
| Adeeb Gaibie        | GBXADE002      |
| Dennis Hammerschlag | HMMDEN001      |
| Hamza Amir          | AMRHAM001      |

| Staff Member              | Department       |
| ------------------------- | ---------------- |
| A/Prof Deshendran Moodley | Computer Science |

# Requirements

python3

See `requirements_all.txt`

# Installation

* First create a new virtual environment by running the following command:

'python3.8 -m venv myenv'

* Activate the virtual environment.
* The activation command varies depending on your operating system:
* On Windows

'myenv\Scripts\activate'

* On Linux or Mac OS X

'source myenv/bin/activate'

* Install all necessary dependencies and modules in requirements_all.txt

'pip3 install -r requirements_all.txt'

# Experiments:

*Only use the following command for all training, HPO and evaluation experiments with the models. Change the default configuration in the config.yaml file for the intended option of either training or performing HPO or evaluating the models to true.*

python3 main.py

Besides from above method there are 2 other methods to run experiments:

1. Through command line arguments.
- One can run the platform using predefined experiment intructions. An example is shown below:
  
    python3 main.py --mode train_tcn

- There are many possible texperiment instructions that can be used. The list is shown below:

| NN Model & functionality  | Instruction    |
| ------------------------- | -------------- |
| TCN training              | train_tcn      |
| TCN HPO                   | tune_tcn       |
| TCN evaluation            | eval_tcn       |
| GWN training              | train_gwn      |
| GWN HPO                   | tune_gwn       |
| GWN HPO                   | tune_gwn       |
| GWN evaluation            | eval_gwn       |
| CLCRN training            | train_clcrn    |
| CLCRN HPO                 | tune_clcrn     |
| CLCRN evaluation          | eval_clcrn     |
| AGCRN training            | train_agcrn    |
| AGCRN tune                | tune_agcrn     |
| AGCRN evaluation          | eval_agcrn     |
| AST-GCN training          | train_astgcn   |
| AST-GCN HPO               | tune_astgcn    |
| AST-GCN evaluation        | eval_astgcn    |
| Visualisation             | vis            |

  
2. Through the in built menu system.
- This system runs in the case when no cofigurations have been set and no commandline paramenters have been selected.
- The intructions used in the built in menu system is the same as the above command line parameter options listed.

### More details on Training, performing HPO and evaluating Models

***Training Models Using Optimal Hyper-Parameters***

For TCN:
Baseline training across 45 weather stations on [3, 6, 9, 12, 24] hour forecasting horizon.

For GWN:
GWN GNN HPO on [3, 6, 9, 12, 24] hour forecasting horizon.

***Performing Random-Search Hyper-Parameter Optimisation(HPO)***

For TCN:
Baseline HPO across 45 weather stations on 24 hour forecasting horizon.

For GWN:
GWN GNN HPO on 24 hour forecasting horizon.

***Evaluating Models' Performance(MSE, RMSE, MAE, SMAPE)***

For TCN:
Evaluation across 21 weather stations on [3, 6, 9, 12, 24] hour forecasting horizon.

For GWN:
GWN GNN HPO on [3, 6, 9, 12, 24] hour forecasting horizon on each of the 21 weather stations:

## Generate visualisation

In config yalm set:  vis default value to true
Then run:
python3 main.py
or use the other methods specified above

##Link to full codebase on github
https://github.com/CodeWizards123/WEATHER-FLOW/tree/main
