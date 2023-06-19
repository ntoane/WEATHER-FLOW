# Weather-Flow

*Deep Learning pipeline for weather prediction using temporal and spatio-temporal deep learning models.*

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
(3.8.10 is working)

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

* Install all necerssary dependencies and modules in requirements_all.txt

'pip3 install -r requirements_all.txt'

# Experiments :

## Training Models Using Optimal Hyper-Parameters

Baseline HPO across 21 weather stations on [3, 6, 9, 12, 24] hour forecasting horizon:

'python3 main.py --train_tcn=True'

GWN GNN HPO on [3, 6, 9, 12, 24] hour forecasting horizon:

'python3 main.py --train_gwn=True'

## Random-Search Hyper-Parameter Optimisation(HPO)

Baseline HPO across 21 weather stations on 24 hour forecasting horizon:

'python3 main.py --tune_tcn=True'

GWN GNN HPO on 24 hour forecasting horizon:

'python3 main.py --tune_gwn=True'

## Evaluate Models' Performance(MSE, RMSE, MAE, SMAPE)

Evaluation across 21 weather stations on [3, 6, 9, 12, 24] hour forecasting horizon:

'python3 main.py --eval_tcn=True'

GWN GNN HPO on [3, 6, 9, 12, 24] hour forecasting horizon on each of the 21 weather stations:

'python3 main.py --eval_gwn=True'
