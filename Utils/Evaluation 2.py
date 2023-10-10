import numpy as np
import os
import pandas as pd
import Utils.metrics as metrics
import Utils.gwnUtils as utils
import Utils.metrics as metrics
import Utils.sharedUtils as sharedUtils
from Logs.modelLogger import modelLogger
import Utils.agcrnUtils as agcrnUtil
import Utils.astgcnUtils as astgcnUtils
     
def TcnEval(tcnConfig, sharedConfig):
    stations = sharedConfig['stations']['default']
    horizons = sharedConfig['horizons']['default']
    num_splits = sharedConfig['n_split']['default']

    for station in stations:
        for split in range(num_splits):
            for horizon in horizons: 
                # try:
                    print(f'TCN evaluation started at {station} for the horizon of {horizon} split {split}')
                    paths = get_tcn_file_paths(station, horizon, split)
                    tcn_logger = modelLogger('tcn', str(station),'Logs/TCN/Evaluation/' + str(horizon) + ' Hour Forecast/'+'tcn_' + str(station) +'.txt', log_enabled=True)
                    tcn_logger.info('TCN evaluation started at' + str(station)+' for the horizon of ' +str(horizon)+ ' for split ' + str(split))
                    # Set the file paths for predictions, targets, and metrics
                    for path in paths.values():
                        sharedUtils.create_file_if_not_exists(path)
                    # Calculate actual vs predicted and metrics using the calculate_tcn_metrics function & save it
                    actual_vs_predicted, metrics = calculate_tcn_metrics(paths)
                    tcn_logger.info('actual vs prediced is :' + str(actual_vs_predicted))
                    tcn_logger.info('saved to file :' +str(paths['actual_vs_predicted_file']) )
                    actual_vs_predicted.to_csv(paths['actual_vs_predicted_file'], index=False)
                    # Write the metrics to the metric file
                    with open(paths['metric_file'], 'w') as metric_file:
                        for name, value in metrics.items():
                            metric_file.write(f'This is the {name}: {value}\n')
                            tcn_logger.info(f'This is the {name}: {value}\n')
                    tcn_logger.info('TCN evaluation of ' + station+' for the horizon of ' +str(horizon) +' was saved to Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt') 
                    print_metrics(metrics, station, horizon)
                # except Exception as e:
                #     print('Error! : Unable to read data or write metrics for station {} and horizon length {}'.format(station, horizon), e)
                #     tcn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}.'.format(station, horizon))
        tcn_logger.info('Finished evaluation of TCN error metrics for all stations.') 


def smape_std(actual, predicted):
        """
        Calculates the standard deviation of SMAPE values
        Parameters:
            actual - target values
            predicted - output values predicted by model
        Returns:
            std - returns the standard deviation of SMAPE values
        """
        smapes = abs(predicted - actual) / ((abs(predicted) + abs(actual)) / 2) * 100
        return np.std(smapes)

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
    smape_std_dev = smape_std(targets.values, preds.values)
    # Compile metrics into a dictionary
    calculated_metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "smape_std_dev": smape_std_dev
    }
    return actual_vs_predicted, calculated_metrics

def GwnEval(gwnConfig, sharedConfig):
    stations = sharedConfig['stations']['default']
    horizons = sharedConfig['horizons']['default']
    num_splits = sharedConfig['n_split']['default']
    s = -1
    # Iterate over each station
    for station in stations:
        # Iterate over each forecasting horizon
        s = s + 1
        for horizon in horizons:
            # try:
                pred = []
                real = []
                gwn_logger = modelLogger('gwn', station,'Logs/GWN/Evaluation/'+'gwn_' + station +'.txt', log_enabled=True)  
                gwn_logger.info('GWN evaluation started at' + station+' for the horizon of ' +str(horizon) ) 
                # Read predictions and targets for each split and append them to pred and real lists
                for split in range(num_splits):
                    print(f'GWN evaluation started at {station} for the horizon of {horizon}')
                    paths = get_gwn_file_paths(station, horizon, split)
                    # Set the file paths for predictions, targets, and metrics
                    for path in paths.values():
                        sharedUtils.create_file_if_not_exists(path)
                    metric_file = paths['metric_file']
                    
                    # Calculate actual vs predicted and metrics using the calculate_gwn_metrics function
                    metrics = calculate_gwn_metrics(pred, real, paths, sharedConfig, gwnConfig, s, horizon)
                    # Save to a text file
                    # actual_vs_predicted.to_csv(paths['actual_vs_predicted_file'], index=False)
                    
                    # Open metric_file for writing
                    with open(metric_file, 'w') as file:
                        # Print and write metrics
                        print_metrics(metrics, station, horizon)
                        # Write the metrics to the metric file
                        for name, value in metrics.items():
                            file.write(f'This is the {name}: {value}\n')
                        gwn_logger.info('Finished computing evaluation error metrics.')
            # except Exception as e:
            #     print('Error! : Unable to read data or write metrics for station {} and forecast length {}.'.format(station, horizon),e)
            #     gwn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}.'.format(station, horizon))
    gwn_logger.info('Finished evaluation of GWN error metrics for all stations.')


def calculate_gwn_metrics(pred, real, paths, sharedConfig, gwnConfig, s, horizon):
    # Read the predictions and targets from the CSV files
    preds = pd.read_pickle(paths['results_file'])
    targets = pd.read_pickle(paths['targets_file'])
    # Create a DataFrame of actual vs predicted values
    # actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
    
    yhat = utils.load_pickle(paths['results_file'])
    target = utils.load_pickle(paths['targets_file'])
    pred = np.append(pred, np.array(yhat).flatten())
    real = np.append(real, np.array(target).flatten()) 
    # Reshape pred and real arrays
    pred = np.array(pred).reshape((int(len(real) / (sharedConfig['n_stations']['default'] * horizon)), 
                                    sharedConfig['n_stations']['default'], horizon))
    real = np.array(real).reshape((int(len(real) / (sharedConfig['n_stations']['default'] * horizon)), 
                                    sharedConfig['n_stations']['default'], horizon))
    # Calculate metrics
    metricsDict = {}
    metricsDict['rmse'] =  metrics.rmse(real[:, s, :], pred[:, s, :])
    metricsDict['mse'] = metrics.mse(real[:, s, :], pred[:, s, :])
    metricsDict['mae'] = metrics.mae(real[:, s, :], pred[:, s, :])
    metricsDict['smape'] = metrics.smape(real[:, s, :], pred[:, s, :])
    
    return metricsDict

def print_metrics(metrics, station, horizon):
    """
    Print evaluation metrics.
    """
    print(f'SMAPE: {metrics["smape"]} at the {station} station forecasting {horizon} hours ahead.')
    print(f'MSE: {metrics["mse"]} at the {station} station forecasting {horizon} hours ahead.')
    print(f'MAE: {metrics["mae"]} at the {station} station forecasting {horizon} hours ahead.')
    print(f'RMSE: {metrics["rmse"]} at the {station} station forecasting {horizon} hours ahead.')
    print('')
     
def get_tcn_file_paths(station, horizon, split, model='TCN'):
    return {
            "yhat_path" : f'Results/TCN/{horizon} Hour Forecast/{station}/Predictions/result_{split}.csv',
            "target_path" : f'Results/TCN/{horizon} Hour Forecast/{station}/Targets/target_{split}.csv',
            "metric_file" : f'Results/TCN/{horizon} Hour Forecast/{station}/Metrics/metrics_{split}.txt',
            "actual_vs_predicted_file" : f'Results/TCN/{horizon} Hour Forecast/{station}/Metrics/actual_vs_predicted.txt'
        }
def get_gwn_file_paths(station, horizon, split,model='GWN'):
    folder_name = f'{horizon} Hour Forecast'
    station_with_spaces = station.replace('_', ' ')
    return {        
        "results_file" : f'Results/{model}/{folder_name}/Predictions/outputs_{split}.pkl',
        "targets_file" : f'Results/{model}/{folder_name}/Targets/targets_{split}.pkl',
        "metric_file" : f'Results/{model}/{folder_name}/Metrics/{station_with_spaces}/metrics_{split}.txt',
        # "actual_vs_predicted_file" : f'Results/{model}/{folder_name}/Metrics/{station_with_spaces}/actual_vs_predicted.txt'
    }
        
def AgcrnEval(modelConfig,sharedConfig):
        stations = sharedConfig['stations']['default'] 
        for horizon in sharedConfig['horizons']['default']:
            for k in range(sharedConfig['n_split']['default']):
                fileDictionary = {
                    'predFile': './Results/AGCRN/' + str(horizon) + ' Hour Forecast/Predictions/outputs_' + str(k),
                    'targetFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Targets/targets_' + str(k),
                    'trainLossFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                    'validationLossFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                    'modelFile': 'Garage/Final Models/AGCRN/' + str(horizon) + ' Hour Models/model_split_' + str(k) + ".pth",
                    'matrixFile': 'Results/AGCRN/' + str(horizon) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                    'metricFile0': './Results/AGCRN/'+  str(horizon)+ ' Hour Forecast/Metrics/',
                    
                    'metricFile1': '/split_' + str(k) + '_metrics.txt'
                }
            
                y_pred=np.load(fileDictionary["predFile"] + ".npy")
                y_true=np.load(fileDictionary["targetFile"] + ".npy")
                
                for i in range(45):
                    station_pred = y_pred[:, :, i, 0]
                    station_true = y_true[:, :, i, 0]
                    print("Evaluating horizon:"+ str(horizon) + " split:" + str(k) + " for station:" + stations[i])
                    # print(station_pred)

                    mae, rmse, mape, _, _ = agcrnUtil.All_Metrics(station_pred, station_true, modelConfig['mae_thresh']['default'], modelConfig['mape_thresh']['default'])

                    filePath =fileDictionary['metricFile0'] +str(stations[i])
                    if not os.path.exists(filePath):
                        os.makedirs(filePath)

                    with open(filePath + fileDictionary['metricFile1'], 'w') as file:
                
                        file.write('This is the MAE ' + str(mae) + '\n')
                        file.write('This is the RMSE ' + str(rmse) + '\n')
                        file.write('This is the MAPE ' + str(mape) + '\n')
        
def evalASTGCN(config, sharedConfig):
    for station in sharedConfig['stations']['default'] :
        for horizon in sharedConfig['horizons']['default']:
    # for station in stations:
    #     for horizon in horizons:
            print(f'ASTGCN evaluation started at {station} for the horizon of {horizon}')
            logger = modelLogger('ASTGCN', str(station),'Logs/ASTGCN/Eval/' + str(horizon) + ' Hour Forecast/'+str(station) +'/'+'astgcn_' + str(station) + '.txt' , log_enabled=False)
            logger.info("ASTGCN evaluation for single time-step started at {station} for the horizon of {horizon}")
            paths = astgcnUtils.get_file_paths(station, horizon)
            try:
                for path in paths.values():
                    astgcnUtils.create_file_if_not_exists(path)
                # Read the predictions, targets & actual vs predicted from the CSV files
                preds = pd.read_csv(paths['yhat']).drop(['Unnamed: 0'], axis=1)
                targets = pd.read_csv(paths['target']).drop(['Unnamed: 0'], axis=1)
                actual_vs_predicted = pd.DataFrame({'Actual': targets.values.flatten(), 'Predicted': preds.values.flatten()})
                actual_vs_predicted.to_csv(paths['actual_vs_predicted'], index=False)
                # Calculate the metrics &  Write the metrics to the files
                metrics = {
                    'MSE': astgcnUtils.MSE(targets.values, preds.values),
                    'RMSE': astgcnUtils.RMSE(targets.values, preds.values),
                    'MAE': astgcnUtils.MAE(targets.values, preds.values),
                    'SMAPE': astgcnUtils.SMAPE(targets.values, preds.values) ,
                    'std_dev_smape': astgcnUtils.smape_std(targets.values, preds.values) 
                }
                with open(paths['metrics'], 'w') as metric_file:
                    for name, value in metrics.items():
                        metric_file.write(f'This is the {name}: {value}\n')
                print(f'ASTGCN evaluation done at {station} for the horizon of {horizon}')
                print(f'And was saved to {paths["metrics"]}')
                logger.info('ASTGCN evaluation done at {station} for the horizon of {horizon}')
                logger.info('And was saved to {paths["metrics"]}')
                for name, value in metrics.items():
                    print(f'{name}: {value} at the {station} station forecasting {horizon} hours ahead.')
                    logger.info('{name}: {value} at the {station} station forecasting {horizon} hours ahead.')
            except Exception as e:
                print(f'Error! Unable to read data or write metrics: {str(e)}')
                logger.error('Error! Unable to read data or write metrics: {str(e)}')
    print('Finished evaluation of ASTGCN error metrics for all stations.')
    logger.info('Finished evaluation of ASTGCN error metrics for all stations')
        
        
        
        
        
        
        
        
        
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
        #             print('  Error! : Unable to read data or write metrics for station {} and forecast length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
        #             gwn_logger.error('Error! : Unable to read data or write metrics for station {} and horizon length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
        # gwn_logger.info('baselineEval : Finished evaluation of GWN error metrics for all stations.')                       