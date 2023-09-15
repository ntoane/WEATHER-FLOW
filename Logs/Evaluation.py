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
from collections import defaultdict
import torch
from Utils.CLCRN_Utils.loss import masked_mse_loss, masked_smape_loss
from Utils.CLCRN_Utils import utils
import pickle
from pathlib import Path

     
def TcnEval(tcnConfig, sharedConfig):
    stations = sharedConfig['stations']['default']
    horizons = sharedConfig['horizons']['default']
    num_splits = sharedConfig['n_split']['default']

    for split in range(num_splits):
        for horizon in horizons: 
            accumulated_metrics = {
                'smape': 0,
                'mse': 0,
                'mae': 0,
                'rmse': 0,
            }    
            
            for station in stations:
            
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
                    # actual_vs_predicted.to_csv(paths['actual_vs_predicted_file'], index=False)
                    # Write the metrics to the metric file
                    with open(paths['metric_file'], 'w') as metric_file:
                        for name, value in metrics.items():
                            metric_file.write(f'This is the {name}: {value}\n')
                            tcn_logger.info(f'This is the {name}: {value}\n')

                            if name in accumulated_metrics:
                                accumulated_metrics[name] += value


                         
                         
                            
        
                    tcn_logger.info('TCN evaluation of ' + station+' for the horizon of ' +str(horizon) +' was saved to Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt') 
                    print_metrics(metrics, station, horizon)

            with open(paths['avr_metrics'], 'w') as metric_file:
                for name, value in accumulated_metrics.items():
                    metric_file.write(f'This is the accumulative {name}: {value/45}\n')


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
                    metricAvr_file =  paths['metricAvr_file']
                    
                    # Calculate actual vs predicted and metrics using the calculate_gwn_metrics function
                    metrics, metricsAvr = calculate_gwn_metrics(pred, real, paths, sharedConfig, gwnConfig, s, horizon)
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

                    with open(metricAvr_file, 'w') as file:
                        # Print and write metrics
                        print_metrics(metricsAvr, station, horizon)
                        # Write the metrics to the metric file
                        for name, value in metricsAvr.items():
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


    metricsDictAvr = {}
    metricsDictAvr['rmse'] =  metrics.rmse(real, pred)
    metricsDictAvr['mse'] = metrics.mse(real, pred)
    metricsDictAvr['mae'] = metrics.mae(real, pred)
    metricsDictAvr['smape'] = metrics.smape(real, pred)


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

    
    return metricsDict, metricsDictAvr

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
            "actual_vs_predicted_file" : f'Results/TCN/{horizon} Hour Forecast/{station}/Metrics/actual_vs_predicted.txt',
            "avr_metrics" :  f'Results/TCN/{horizon} Hour Forecast/average/avr_metrics_{split}.txt'
        }
def get_gwn_file_paths(station, horizon, split,model='GWN'):
    folder_name = f'{horizon} Hour Forecast'
    station_with_spaces = station.replace('_', ' ')
    return {        
        "results_file" : f'Results/{model}/{folder_name}/Predictions/outputs_{split}.pkl',
        "targets_file" : f'Results/{model}/{folder_name}/Targets/targets_{split}.pkl',
        "metric_file" : f'Results/{model}/{folder_name}/Metrics/{station_with_spaces}/metrics_{split}.txt',
        "metricAvr_file" : f'Results/{model}/{folder_name}/Metrics/average/metrics_{split}.txt',
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
                
                def read_value_from_file(filename):
                    with open(filename, 'r') as file:
                        return float(file.read().strip())

                train_data_min_file = './Results/AGCRN/' + str(horizon) + ' Hour Forecast/scaler/min_' + str(k) + ".csv"
                train_data_max_file = './Results/AGCRN/' + str(horizon) + ' Hour Forecast/scaler/max_' + str(k) + ".csv"

                min_scalar = read_value_from_file(train_data_min_file)
                max_scalar = read_value_from_file(train_data_max_file)

                y_pred=np.load(fileDictionary["predFile"] + ".npy")
                y_true=np.load(fileDictionary["targetFile"] + ".npy")


                # # scaler = utils.NormScaler(0, 43)
                # # scaler = utils.NormScaler(y_trueO.min(), y_trueO.max())
                # scaler = utils.NormScaler(min_scalar, max_scalar)
                # y_true = scaler.transform(y_trueO)
                # y_pred = scaler.transform(y_predO)

                #per station metrics
                for i in range(45):
                    station_pred = y_pred[:, :, i, 0]
                    station_true = y_true[:, :, i, 0]
                    # print(station_true)
                    print("Evaluating horizon:"+ str(horizon) + " split:" + str(k) + " for station:" + stations[i])
                    # print(station_pred)

                    rmse =  metrics.rmse(station_true, station_pred)
                    mse = metrics.mse(station_true, station_pred)
                    mae = metrics.mae(station_true, station_pred)
                    smape = metrics.smape(station_true.flatten(), station_pred.flatten())


                    filePath =fileDictionary['metricFile0'] +str(stations[i])
                    if not os.path.exists(filePath):
                        os.makedirs(filePath)

                    with open(filePath + fileDictionary['metricFile1'], 'w') as file:
                        file.write('This is the RMSE ' + str(rmse) + '\n')
                        file.write('This is the MSE ' + str(mse) + '\n')
                        file.write('This is the MAE ' + str(mae) + '\n')
                        file.write('This is the SMAPE ' + str(smape) + '\n')


                #all station collective metrics
                y_true=y_true.flatten()
                y_pred=y_pred.flatten()
                rmse =  metrics.rmse(y_true, y_pred)
                mse = metrics.mse(y_true, y_pred)
                mae = metrics.mae(y_true, y_pred)
                smape = metrics.smape(y_true, y_pred)

                filePath =fileDictionary['metricFile0'] +"average"
                if not os.path.exists(filePath):
                    os.makedirs(filePath)

                with open(filePath + fileDictionary['metricFile1'], 'w') as file:
                    file.write('This is the RMSE ' + str(rmse) + '\n')
                    file.write('This is the MSE ' + str(mse) + '\n')
                    file.write('This is the MAE ' + str(mae) + '\n')
                    file.write('This is the SMAPE ' + str(smape) + '\n')



def AstgcnEval(config, sharedConfig):
    stations = sharedConfig['stations']['default']
    n_splits = sharedConfig['n_split']['default']
    all_metrics = {}  # To store all metrics for each station
    avg_metrics = {}
    all_smape_values = []  # List to collect all SMAPE values

    for horizon in sharedConfig['horizons']['default']:
        print("Evaluating horizon : ",horizon)
        for k in range(n_splits):
            fileDictionary = {
                "yhat": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Predictions/result.csv',
                "target": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Targets/target.csv',
                "metrics": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/',
                'metricFile1': '/split_' + str(k) + '_metrics.txt', 
                "actual_vs_predicted": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/actual_vs_predicted.txt'
            }

            y_pred = pd.read_csv(fileDictionary["yhat"])['0'].values
            y_true = pd.read_csv(fileDictionary["target"])['0'].values   
            y_pred_reshaped = y_pred.reshape(-1, 45)
            y_true_reshaped = y_true.reshape(-1, 45)

            for i in range(45):
                station_pred = y_pred_reshaped[:, i]
                station_true = y_true_reshaped[:, i]
                
                mse = metrics.mse(np.array([station_true]), np.array([station_pred]))
                rmse = metrics.rmse(np.array([station_true]), np.array([station_pred]))
                mae = metrics.mae(np.array([station_true]), np.array([station_pred]))
                smape = metrics.smape(np.array([station_true]), np.array([station_pred]))
                print(f"SMAPE for station {stations[i]}: {smape}")


                if stations[i] not in all_metrics:
                    all_metrics[stations[i]] = {'mse': [], 'rmse': [], 'mae': [], 'smape': []}
                
                all_metrics[stations[i]]['mse'].append(mse)
                all_metrics[stations[i]]['rmse'].append(rmse)
                all_metrics[stations[i]]['mae'].append(mae)
                all_metrics[stations[i]]['smape'].append(smape)

                filePath = fileDictionary['metrics'] + str(stations[i])
                if not os.path.exists(filePath):
                    os.makedirs(filePath)

                with open(filePath + fileDictionary['metricFile1'], 'w') as file:
                    file.write('This is the RMSE ' + str(rmse) + '\n')
                    file.write('This is the MSE ' + str(mse) + '\n')
                    file.write('This is the MAE ' + str(mae) + '\n')
                    file.write('This is the SMAPE ' + str(smape) + '\n')

    for station in stations:
        avg_metrics[station] = {}
        for metric in ['mse', 'rmse', 'mae', 'smape']:
            avg_value = np.mean(all_metrics[station][metric])
            avg_metrics[station][metric] = avg_value

            if metric == 'smape':
                all_smape_values.append(avg_value)

            best_metric_file_path = f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/average_metrics_{station}.txt'
            with open(best_metric_file_path, 'a') as file:
                file.write(f'Average {metric.upper()}: {avg_value}\n')

    summary_file_path = f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/top_bottom_summary.txt'
    with open(summary_file_path, 'w') as file:
        for metric in ['smape']:
            sorted_stations = sorted(stations, key=lambda x: avg_metrics[x][metric])
            top_5_stations = sorted_stations[:5]
            bottom_5_stations = sorted_stations[-5:]

            file.write(f'Top 5 stations for {metric.upper()}:\n')
            for station in top_5_stations:
                file.write(f'{station}: {avg_metrics[station][metric]}\n')
            
            file.write(f'\nBottom 5 stations for {metric.upper()}:\n')
            for station in bottom_5_stations:
                file.write(f'{station}: {avg_metrics[station][metric]}\n')
            file.write('\n')

    global_avg_smape = np.mean(all_smape_values)
    print(f"Global Average SMAPE across all stations: {global_avg_smape}")

    with open(summary_file_path, 'a') as file:
        file.write(f'\nGlobal Average SMAPE across all stations: {global_avg_smape}\n')
        
        

def ClcrnEval(sharedConfig,modelConfig):
       
        print("Evaluating model per station")
        # print()

        h =  sharedConfig['horizons']['default']

        log_dir = get_log_dir(sharedConfig,modelConfig)
        
        true_file = '{}/actuals.pkl'.format(log_dir,h)
        predict_file = '{}/predictions.pkl'.format(log_dir,h)
        # Open the pickle file for reading (rb mode)
        with open(true_file, 'rb') as f:
            # Load the content of the pickle file
            trueVals = pickle.load(f)
        with open(predict_file, 'rb') as f:
            # Load the content of the pickle file
            predVals = pickle.load(f)

        print("Average smape value for horizon {}".format(h[0]))
        print(masked_smape_loss(torch.tensor(predVals['y_preds']),torch.tensor(trueVals['y_trues'])))

        trues = torch.tensor(trueVals['y_trues'])
        preds = torch.tensor(predVals['y_preds'])
        smape = 0
        mse = 0
        
        fpath = '{}/stationScore_{}.txt'.format(modelConfig['results_dir']['default'],h[0])
        with open(fpath, "w") as f:
            f.write("Metrics scores at each station  \n")
            for i in range(modelConfig['num_nodes']['default']):
                t = trues[:,:,:i+1,:]
                p = preds[:,:,:i+1,:]
                score = masked_smape_loss(p,t)
                mscore = masked_mse_loss(p,t)
                smape = smape + score
                mse = mse + mscore

                
                smapeData = 'SMAPE score at station {} : {} \n'.format(i+1,score.item())
                mseData = 'MSE score at station {} : {} \n'.format(i+1,mscore.item())
               
                f.write(smapeData)
                f.write(mseData)

                print(smapeData)

            f.write("Average SMAPE maually calculated: {} \n".format(smape/45))
            f.write("Average MSE maually calculated: {}".format(mse/45))

        print("Average SMAPE maually calculated: {}".format(smape/45))
        print("Average MSE maually calculated: {}".format(mse/45))        

def get_log_dir(sharedConfig,modelConfig):
        log_dir = Path(modelConfig['log_dir']['default'])/'{} Hour Forecast'.format(sharedConfig['horizons']['default'][0])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir            