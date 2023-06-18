import pandas as pd
import Utils.metrics as metrics
import Utils.gwnUtils as utils
import Utils.metrics as metrics
import numpy as np

import logging
from Evaluation.modelLogger import modelLogger


def TcnEval(stations, model):
    """
    Calculates the LSTM/TCN model's performance on the test set for each station. These metrics are written to a file
    for each station. The predictions are read from the results file for each station. The targets are pulled from
    the weather stations' data sets. The MSE, MAE, RMSE, and SMAPE metrics are calculated on all forecasting
    horizons(3, 6, 9, 12, and 24) for each individual weather station. The metrics for each station, across all
    forecasting horizons are then written to a text file.

    Parameters:
        stations - List of the weather stations.
        model - Whether these metrics are being calculated for the LSTM or TCN model
    """

    for station in stations:
        for horizon in [3, 6, 9, 12, 24]:
            try:
                # Set the file paths for predictions, targets, and metrics
                yhat_path = f'Results/{model}/{horizon} Hour Forecast/{station}/Predictions/result.csv'
                target_path = f'Results/{model}/{horizon} Hour Forecast/{station}/Targets/target.csv'
                metric_file = f'Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt'
                # Read the predictions and targets from the CSV files
                preds = pd.read_csv(yhat_path).drop(['Unnamed: 0'], axis=1)
                targets = pd.read_csv(target_path).drop(['Unnamed: 0'], axis=1)

                # Calculate the metrics
                mse = metrics.mse(targets.values, preds.values)
                rmse = metrics.rmse(targets.values, preds.values)
                mae = metrics.mae(targets.values, preds.values)
                smape = metrics.smape(targets.values, preds.values)

                # Write the metrics to the metric file
                with open(metric_file, 'w') as metric:
                    metric.write('This is the MSE: {}\n'.format(mse))
                    metric.write('This is the MAE: {}\n'.format(mae))
                    metric.write('This is the RMSE: {}\n'.format(rmse))
                    metric.write('This is the SMAPE: {}\n'.format(smape))
                # Print the metrics for the current station and horizon length
                print('SMAPE: {0} at the {1} station forecasting {2} hours ahead.'.format(smape, station, horizon))
                print('MSE: {0} at the {1} station forecasting {2} hours ahead.'.format(mse, station, horizon))
                print('MAE: {0} at the {1} station forecasting {2} hours ahead.'.format(mae, station, horizon))
                print('RMSE: {0} at the {1} station forecasting {2} hours ahead.'.format(rmse, station, horizon))
                print('')
            except IOError:
                print('Error! : Unable to read data or write metrics for station {} and horizon length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
                
                
def GwnEval(stations, args):
    """
     Calculates the GWN model's performance on the test set across all forecasting horizons [3, 6, 9, 12, 24] for each
     individual station. The predictions are read from the results file for each split of the walk-forward validation
     method. The predictions from each split are appended into one long list of predictions. The targets are pulled from
     targets file in the GWN results directory. The MSE, MAE, RMSE, and SMAPE metrics are then calculated and written
     to the metric files.

     Parameters:
         stations - List of the weather stations.
         args - Parser of parameter arguments.
     """
    num_splits = 27
    num_stations = 21
    gwn_logger = modelLogger('gwn', 'Evaluation/Logs/GWN/gwn_logs.txt')
    gwn_logger.info('baselineEval : Starting to compute evaluation error metrics.')

    # Iterate over each station
    for station in range(num_stations):
        # Iterate over each forecasting horizon
        for horizon in [3, 6, 9, 12, 24]:
            try:
                pred = []
                real = []
                # Read predictions and targets for each split and append them to pred and real lists
                for split in range(num_splits):
                    results_file = f'Results/GWN/{horizon} Hour Forecast/Predictions/outputs_{split}.pkl'
                    targets_file = f'Results/GWN/{horizon} Hour Forecast/Targets/targets_{split}.pkl'
                    metric_file = f'Results/GWN/Metrics/{stations[station]}/metrics_{horizon}'
                    
                    yhat = utils.load_pickle(results_file)
                    target = utils.load_pickle(targets_file)
                    pred.extend(np.array(yhat).flatten())
                    real.extend(np.array(target).flatten())
                
                    # Reshape pred and real arrays
                    pred = np.array(pred).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
                                                args.seq_length))
                    real = np.array(real).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
                                                args.seq_length))

                    # Open metric_file for writing
                    with open(metric_file, 'w') as file:
                        preds = pred[:, station, :]
                        real_values = real[:, station, :]
                        # Calculate metrics
                        root = metrics.rmse(real_values, preds)
                        square = metrics.mse(real_values, preds)
                        abs_val = metrics.mae(real_values, preds)
                        ape = metrics.smape(real_values, preds)
                        # Print and write metrics
                        print('RMSE: {0} for station {1} forecasting {2} hours ahead'.format(root, station+1, horizon))
                        print('MSE: {0} for station {1} forecasting {2} hours ahead'.format(square, station+1, horizon))
                        print('MAE: {0} for station {1} forecasting {2} hours ahead'.format(abs_val, station+1, horizon))
                        print('SMAPE: {0} for station {1} forecasting {2} hours ahead'.format(ape, station+1, horizon))
                        print('')
                        file.write('This is the MSE ' + str(square) + '\n')
                        file.write('This is the MAE ' + str(abs_val) + '\n')
                        file.write('This is the RMSE ' + str(root) + '\n')
                        file.write('This is the SMAPE ' + str(ape) + '\n')
                        gwn_logger.info('baselineEval : Finished computing evaluation error metrics.')
            except IOError:
                #metric_file = f'Results/GWN/Metrics/{stations[station]}/metrics_{horizon}'
                #print(f'Error: Unable to write metrics to {metric_file}')
                print('Error! : Unable to read data or write metrics for station {} and forecast length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
                
                
# Will be removed in the future
# def eval(stations, model):
#     """
#     Calculates the LSTM/TCN model's performance on the test set for each station. These metrics are written to a file
#     for each station. The predictions are read from the results file for each station. The targets are pulled from
#     the weather stations' data sets. The MSE, MAE, RMSE, and SMAPE metrics are calculated on all forecasting
#     horizons(3, 6, 9, 12, and 24) for each individual weather station. The metrics for each station, across all
#     forecasting horizons are then written to a text file.

#     Parameters:
#         stations - List of the weather stations.
#         model - Whether these metrics are being calculated for the LSTM or TCN model
#     """

#     for station in stations:
#         for forecast_len in [3, 6, 9, 12, 24]:
#             yhat = 'Results/' + model + '/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
#                    'result.csv'
#             target = 'Results/' + model + '/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' + \
#                      'target.csv'
#             metricFile = 'Results/' + model + '/' + str(forecast_len) + ' Hour Forecast/' + station + '/Metrics/' + \
#                          '/metrics.txt'

#             preds = pd.read_csv(yhat)
#             preds = preds.drop(['Unnamed: 0'], axis=1)
#             metric = open(metricFile, 'w')

#             targets = pd.read_csv(target)
#             targets = targets.drop(['Unnamed: 0'], axis=1)

#             preds = preds.values
#             targets = targets.values

#             mse = metrics.mse(targets, preds)
#             rmse = metrics.rmse(targets, preds)
#             mae = metrics.mae(targets, preds)
#             SMAPE = metrics.smape(targets, preds)

#             metric.write('This is the MSE ' + str(mse) + '\n')
#             metric.write('This is the MAE ' + str(mae) + '\n')
#             metric.write('This is the RMSE ' + str(rmse) + '\n')
#             metric.write('This is the SMAPE ' + str(SMAPE) + '\n')

#             print('SMAPE: {0} at the {1} station forecasting {2} hours ahead. '.format(SMAPE, station, forecast_len))
#             print('MSE: {0} at the {1} station forecasting {2} hours ahead. '.format(mse, station, forecast_len))
#             print('MAE: {0} at the {1} station forecasting {2} hours ahead. '.format(mae, station, forecast_len))
#             print('RMSE: {0} at the {1} station forecasting {2} hours ahead. '.format(rmse, station, forecast_len))
#             print('')
#             metric.close()