import os
import numpy as np
import pandas as pd
import Utils.metrics as metrics
import Utils.gwnUtils as utils
import Utils.metrics as metrics
import Utils.sharedUtils as sharedUtils
from Logs.modelLogger import modelLogger
import pickle
     
def TcnEval(tcnConfig, sharedConfig):
    stations = sharedConfig['stations']['default']
    horizons = sharedConfig['horizons']['default']
    tcn_logger = modelLogger('tcn', 'all','Logs/TCN/Evaluation/tcn_all_stations.txt', log_enabled=False)
    tcn_logger.info('baselineEval : TCN evaluation started at all stations set for evaluation :)')
    for station in stations:
        for horizon in horizons: 
            try:
                print(f'TCN evaluation started at {station} for the horizon of {horizon}')
                paths = get_tcn_file_paths(station, horizon)
                tcn_logger = modelLogger('tcn', str(station),'Logs/TCN/Evaluation/'+'tcn_' + str(station) +'.txt', log_enabled=False)
                tcn_logger.info('baselineEval : TCN evaluation started at' + str(station)+' for the horizon of ' +str(horizon))
                # Set the file paths for predictions, targets, and metrics
                for path in paths.values():
                    sharedUtils.create_file_if_not_exists(path)
                # Calculate actual vs predicted and metrics using the calculate_tcn_metrics function & save it
                actual_vs_predicted, metrics = calculate_tcn_metrics(paths)
                actual_vs_predicted.to_csv(paths['actual_vs_predicted_file'], index=False)
                # Write the metrics to the metric file
                with open(paths['metrics'], 'w') as metric_file:
                    for name, value in metrics.items():
                        metric_file.write(f'This is the {name}: {value}\n')
                tcn_logger.info('baselineEval : TCN evaluation of ' + station+' for the horizon of ' +str(horizon) +' was saved to Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt') 
                print_metrics(metrics, station, horizon)
            except Exception as e:
                print('Error! : Unable to read data or write metrics for station {} and horizon length {}'.format(station, horizon), e)
                tcn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}.'.format(station, horizon))
    tcn_logger.info('baselineEval : Finished evaluation of TCN error metrics for all stations.') 


def calculate_tcn_metrics(paths):
    # Read the predictions and targets from the CSV files, pkl type files
    preds = pd.read_csv(paths['yhat_path']).drop(['Unnamed: 0'], axis=1)
    targets = pd.read_csv(paths['target_path']).drop(['Unnamed: 0'], axis=1)

    # Create a DataFrame of actual vs predicted values
    actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
    
    # Calculate the metrics
    mse = metrics.mse(targets.values, preds.values)
    rmse = metrics.rmse(targets.values, preds.values)
    mae = metrics.mae(targets.values, preds.values)
    smape = metrics.smape(targets.values, preds.values)
    
    # Compile metrics into a dictionary
    calculated_metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "smape": smape
    }
    
    return actual_vs_predicted, calculated_metrics


def GwnEval(gwnConfig, sharedConfig):
    stations = sharedConfig['stations']['default']
    horizons = sharedConfig['horizons']['default']
    num_splits = sharedConfig['n_split']['default']
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
                    print(f'GWN evaluation started at {station} for the horizon of {horizon}')
                    paths = get_gwn_file_paths(station, horizon, split)
                    # Set the file paths for predictions, targets, and metrics
                    for path in paths.values():
                        sharedUtils.create_file_if_not_exists(path)
                    metric_file = paths['metric_file']
                    
                    # Calculate actual vs predicted and metrics using the calculate_gwn_metrics function
                    actual_vs_predicted, metrics = calculate_gwn_metrics(paths, sharedConfig, gwnConfig, s)
                    # Save to a text file
                    actual_vs_predicted.to_csv(paths['actual_vs_predicted_file'], index=False)
                    
                    # Open metric_file for writing
                    with open(metric_file, 'w') as file:
                        # Print and write metrics
                        print_metrics(metrics, station, horizon)
                        # Write the metrics to the metric file
                        for name, value in metrics.items():
                            file.write(f'This is the {name}: {value}\n')
                        gwn_logger.info('baselineEval : Finished computing evaluation error metrics.')
            except Exception as e:
                print('Error! : Unable to read data or write metrics for station {} and forecast length {}.'.format(station, horizon),e)
                gwn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}.'.format(station, horizon))
    gwn_logger.info('baselineEval : Finished evaluation of GWN error metrics for all stations.')


def calculate_gwn_metrics(paths, sharedConfig, gwnConfig, s):
    # Read the predictions and targets from the CSV files
    preds = pd.read_pickle(paths['results_file'])
    targets = pd.read_pickle(paths['targets_file'])
    # Create a DataFrame of actual vs predicted values
    actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
    
    yhat = utils.load_pickle(paths['results_file'])
    target = utils.load_pickle(paths['targets_file'])
    pred = np.append(pred, np.array(yhat).flatten())
    real = np.append(real, np.array(target).flatten()) 
    # Reshape pred and real arrays
    pred = np.array(pred).reshape((int(len(real) / (sharedConfig['n_stations']['default'] * gwnConfig['seq_length']['default'])), 
                                    sharedConfig['n_stations']['default'],
                                    gwnConfig['seq_length']['default']))
    real = np.array(real).reshape((int(len(real) / (sharedConfig['n_stations']['default'] * gwnConfig['seq_length']['default'])), 
                                    sharedConfig['n_stations']['default'],
                                    gwnConfig['seq_length']['default']))
    # Calculate metrics
    metrics = {}
    metrics['root'] = metrics.rmse(real[:, s, :], pred[:, s, :])
    metrics['square'] = metrics.mse(real[:, s, :], pred[:, s, :])
    metrics['abs_val'] = metrics.mae(real[:, s, :], pred[:, s, :])
    metrics['ape'] = metrics.smape(real[:, s, :], pred[:, s, :])
    
    return actual_vs_predicted, metrics

    
def print_metrics(metrics, station, horizon):
    """
    Print evaluation metrics.
    """
    print(f'SMAPE: {metrics["smape"]} at the {station} station forecasting {horizon} hours ahead.')
    print(f'MSE: {metrics["mse"]} at the {station} station forecasting {horizon} hours ahead.')
    print(f'MAE: {metrics["mae"]} at the {station} station forecasting {horizon} hours ahead.')
    print(f'RMSE: {metrics["rmse"]} at the {station} station forecasting {horizon} hours ahead.')
    print('')
     
def get_tcn_file_paths(station, horizon, model='TCN'):
    return {
            "yhat_path" : f'Results/TCN/{horizon}_Hour_Forecast/{station}/Predictions/result.csv',
            "target_path" : f'Results/TCN/{horizon}_Hour_Forecast/{station}/Targets/target.csv',
            "metric_file" : f'Results/TCN/{horizon}_Hour_Forecast/Metrics/{station}/metrics.txt',
            "actual_vs_predicted_file" : f'Results/TCN/{horizon}_Hour_Forecast/{station}/Metrics/actual_vs_predicted.txt'
        }
       

def get_gwn_file_paths(station, horizon, split,model='GWN'):
    return{        
            "results_file" : f'Results/GWN/{horizon} Hour Forecast/Predictions/outputs_{split}.pkl',
            "targets_file" : f'Results/GWN/{horizon} Hour Forecast/Targets/targets_{split}.pkl',
            # "metric_file_directory" : f'Results/GWN/{horizon}_Hour_Forecast/Metrics/{station}/',
            "metric_file" : f'Results/GWN/{horizon}_Hour_Forecast/Metrics/{station}/metrics_{split}.txt',
            # "metric_filename" : 'metrics.txt',
            "actual_vs_predicted_file" : f'Results/GWN/{horizon} Hour Forecast/{station}/Metrics/actual_vs_predicted.txt'
        }
        
        
        
        
        
        
        
        
        
        
        # Old versions
        # def GwnEval(self,gwnConfig):
        # num_splits = self.sharedConfig['n_split']['default']
        # gwn_logger = modelLogger('gwn','all','Logs/GWN/gwn_all_stations.txt', log_enabled=False)
        # gwn_logger.info('baselineEval : Starting to compute evaluation error metrics for all stations.')
        # s = -1
        # # Iterate over each station
        # for station in self.stations:
        #     # Iterate over each forecasting horizon
        #     s = s + 1
        #     for horizon in self.horizons:
        #         try:
        #             pred = []
        #             real = []
        #             gwn_logger = modelLogger('gwn', station,'Logs/GWN/Evaluation/'+'gwn_' + station +'.txt', log_enabled=False)  
        #             gwn_logger.info('baselineEval : GWN evaluation started at' + station+' for the horizon of ' +str(horizon) ) 
        #             # Read predictions and targets for each split and append them to pred and real lists
        #             for split in range(num_splits):
        #                 results_file = f'Results/GWN/{horizon} Hour Forecast/Predictions/outputs_{split}.pkl'
        #                 targets_file = f'Results/GWN/{horizon} Hour Forecast/Targets/targets_{split}.pkl'
        #                 metric_file_directory = f'Results/GWN/{horizon}_Hour_Forecast/Metrics/{station}/'
        #                 metric_filename = 'metrics.txt'
        #                 actual_vs_predicted_file = f'Results/GWN/{horizon} Hour Forecast/{station}/Metrics/actual_vs_predicted.txt'
        #                 sharedUtils.create_file_if_not_exists(results_file)
        #                 sharedUtils.create_file_if_not_exists(targets_file)
        #                 sharedUtils.create_file_if_not_exists(metric_file_directory)
        #                 sharedUtils.create_file_if_not_exists(actual_vs_predicted_file)
        #                 metric_file = os.path.join(metric_file_directory, metric_filename)
        #                 # Read the predictions and targets from the CSV files
        #                 preds = pd.read_csv(results_file).drop(['Unnamed: 0'], axis=1)
        #                 targets = pd.read_csv(targets_file).drop(['Unnamed: 0'], axis=1)
        #                 # Create a DataFrame of actual vs predicted values
        #                 actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
        #                 # Save to a text file
        #                 actual_vs_predicted.to_csv(actual_vs_predicted_file, index=False)
                        
        #                 gwn_logger = modelLogger('gwn', str(station), 'Logs/GWN/gwn_.txt', log_enabled=False)
        #                 yhat = utils.load_pickle(results_file)
        #                 target = utils.load_pickle(targets_file)
        #                 # pred.extend(np.array(yhat).flatten())
        #                 # real.extend(np.array(target).flatten())
        #                 pred = np.append(pred, np.array(yhat).flatten())
        #                 real = np.append(real, np.array(target).flatten())
                        
        #                 # Reshape pred and real arrays
        #                 pred = np.array(pred).reshape((int(len(real) / (self.sharedConfig['n_stations']['default'] * gwnConfig['seq_length']['default'])), 
        #                                             self.sharedConfig['n_stations']['default'],
        #                                             gwnConfig['seq_length']['default']))
        #                 real = np.array(real).reshape((int(len(real) / (self.sharedConfig['n_stations']['default'] * gwnConfig['seq_length']['default'])), 
        #                                             self.sharedConfig['n_stations']['default'],
        #                                             gwnConfig['seq_length']['default']))
        #                 # Open metric_file for writing
        #                 with open(metric_file, 'w') as file:
        #                     preds = pred[:, s, :]
        #                     real_values = real[:, s, :]
        #                     # Calculate metrics
        #                     root = metrics.rmse(real_values, preds)
        #                     square = metrics.mse(real_values, preds)
        #                     abs_val = metrics.mae(real_values, preds)
        #                     ape = metrics.smape(real_values, preds)
        #                     # Print and write metrics
        #                     print('RMSE: {0} for station {1} forecasting {2} hours ahead'.format(root, station, horizon))
        #                     print('MSE: {0} for station {1} forecasting {2} hours ahead'.format(square, station, horizon))
        #                     print('MAE: {0} for station {1} forecasting {2} hours ahead'.format(abs_val, station, horizon))
        #                     print('SMAPE: {0} for station {1} forecasting {2} hours ahead'.format(ape, station, horizon))
        #                     print('')
        #                     file.write('This is the MSE ' + str(square) + '\n')
        #                     file.write('This is the MAE ' + str(abs_val) + '\n')
        #                     file.write('This is the RMSE ' + str(root) + '\n')
        #                     file.write('This is the SMAPE ' + str(ape) + '\n')
        #                     gwn_logger.info('baselineEval : Finished computing evaluation error metrics.')
        #         except IOError:
        #             #metric_file = f'Results/GWN/Metrics/{stations[station]}/metrics_{horizon}'
        #             #print(f'Error: Unable to write metrics to {metric_file}')
        #             print('Error! : Unable to read data or write metrics for station {} and forecast length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
        #             gwn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
        # gwn_logger.info('baselineEval : Finished evaluation of GWN error metrics for all stations.')                       