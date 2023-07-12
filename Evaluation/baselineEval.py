import pandas as pd
import Utils.metrics as metrics
import Utils.gwnUtils as utils
import Utils.metrics as metrics
import numpy as np
import os

import logging
from Logs.modelLogger import modelLogger


def TcnEval(model, sharedConfig, tcnConfig):
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

    stations = sharedConfig['stations']['default']
    horizons = sharedConfig['horizons']['default']

    tcn_logger = modelLogger('tcn', 'all','Logs/TCN/Evaluation/'+'tcn_all_stations.txt', log_enabled=False)  
    tcn_logger.info('baselineEval : TCN evaluation started at all stations set for evaluation :)') 
    
    for station in stations:
        for horizon in horizons:
            try:
                tcn_logger = modelLogger('tcn', str(station),'Logs/TCN/Evaluation/'+'tcn_' + str(station) +'.txt', log_enabled=False)  
                tcn_logger.info('baselineEval : TCN evaluation started at' + str(station)+' for the horizon of ' +str(horizon) ) 
                
                
                # Set the file paths for predictions, targets, and metrics
                yhat_path = f'Results/{model}/{horizon} Hour Forecast/{station}/Predictions/result.csv'
                target_path = f'Results/{model}/{horizon}Hour Forecast/{station}/Targets/target.csv'
                metric_file = f'Results/{model}/{horizon}_Hour_Forecast/Metrics/{station}/metrics.txt'
                
               
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
                
                tcn_logger.info('baselineEval : TCN evaluation done at' + station+' for the horizon of ' +str(horizon) ) 
                tcn_logger.info('baselineEval : TCN evaluation of ' + station+' for the horizon of ' +str(horizon) +' was saved to Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt' ) 
                
                # Print the metrics for the current station and horizon length
                print('SMAPE: {0} at the {1} station forecasting {2} hours ahead.'.format(smape, station, horizon))
                print('MSE: {0} at the {1} station forecasting {2} hours ahead.'.format(mse, station, horizon))
                print('MAE: {0} at the {1} station forecasting {2} hours ahead.'.format(mae, station, horizon))
                print('RMSE: {0} at the {1} station forecasting {2} hours ahead.'.format(rmse, station, horizon))
                print('')
            except IOError:
                print('Error! : Unable to read data or write metrics for station {} and horizon length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
                tcn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
                
    tcn_logger.info('baselineEval : Finished evaluation of TCN error metrics for all stations.')  
                
def GwnEval(sharedConfig, gwnConfig):
    """
     Calculates the GWN model's performance on the test set across all forecasting horizons [3, 6, 9, 12, 24] for each
     individual station. The predictions are read from the results file for each split of the walk-forward validation
     method. The predictions from each split are appended into one long list of predictions. The targets are pulled from
     targets file in the GWN results directory. The MSE, MAE, RMSE, and SMAPE metrics are then calculated and written
     to the metric files.

     Parameters:
         stations - List of the weather stations.
         config - Conguration file of parameter arguments.
     """
    horizons = sharedConfig['horizons']['default']
    num_splits = sharedConfig['n_split']['default']    #was 27 (made use config)
    stations = sharedConfig['stations']['default']    #was 21 (made us config)
    gwn_logger = modelLogger('gwn','all','Logs/GWN/gwn_all_stations.txt', log_enabled=False)
    gwn_logger.info('baselineEval : Starting to compute evaluation error metrics for all stations.')
    
    s = -1
    # Iterate over each station
    for station in stations:
        # Iterate over each forecasting horizon
        s = s + 1
        for horizon in horizons:
            try:
                pred = []
                real = []
                gwn_logger = modelLogger('gwn', station,'Logs/GWN/Evaluation/'+'gwn_' + station +'.txt', log_enabled=False)  
                gwn_logger.info('baselineEval : GWN evaluation started at' + station+' for the horizon of ' +str(horizon) ) 
                # Read predictions and targets for each split and append them to pred and real lists
                for split in range(num_splits):
                    results_file = f'Results/GWN/{horizon} Hour Forecast/Predictions/outputs_{split}.pkl'
                    targets_file = f'Results/GWN/{horizon} Hour Forecast/Targets/targets_{split}.pkl'

                    metric_file_directory = f'Results/GWN/{horizon}_Hour_Forecast/Metrics/{station}/'
                    metric_filename = 'metrics.txt'
                    if not os.path.exists(metric_file_directory):
                        os.makedirs(metric_file_directory)

                    metric_file = os.path.join(metric_file_directory, metric_filename)




                    
                    gwn_logger = modelLogger('gwn', str(station), 'Logs/GWN/gwn_.txt', log_enabled=False)
                    yhat = utils.load_pickle(results_file)
                    target = utils.load_pickle(targets_file)
                    # pred.extend(np.array(yhat).flatten())
                    # real.extend(np.array(target).flatten())
                    pred = np.append(pred, np.array(yhat).flatten())
                    real = np.append(real, np.array(target).flatten())

                    # Reshape pred and real arrays
                    pred = np.array(pred).reshape((int(len(real) / (sharedConfig['n_stations']['default'] * gwnConfig['seq_length']['default'])), 
                                                sharedConfig['n_stations']['default'],
                                                gwnConfig['seq_length']['default']))
                    real = np.array(real).reshape((int(len(real) / (sharedConfig['n_stations']['default'] * gwnConfig['seq_length']['default'])), 
                                                sharedConfig['n_stations']['default'],
                                                gwnConfig['seq_length']['default']))

                    # Open metric_file for writing
                    with open(metric_file, 'w') as file:
                        preds = pred[:, s, :]
                        real_values = real[:, s, :]
                        # Calculate metrics
                        root = metrics.rmse(real_values, preds)
                        square = metrics.mse(real_values, preds)
                        abs_val = metrics.mae(real_values, preds)
                        ape = metrics.smape(real_values, preds)
                        # Print and write metrics
                        print('RMSE: {0} for station {1} forecasting {2} hours ahead'.format(root, station, horizon))
                        print('MSE: {0} for station {1} forecasting {2} hours ahead'.format(square, station, horizon))
                        print('MAE: {0} for station {1} forecasting {2} hours ahead'.format(abs_val, station, horizon))
                        print('SMAPE: {0} for station {1} forecasting {2} hours ahead'.format(ape, station, horizon))
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
                gwn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
    
    gwn_logger.info('baselineEval : Finished evaluation of GWN error metrics for all stations.')            
                
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