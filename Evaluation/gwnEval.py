import Utils.gwnUtils as utils
import Utils.metrics as metrics
import numpy as np


# def eval(stations, args):
#     """
#      Calculates the GWN model's performance on the test set across all forecasting horizons[3, 6, 9, 12, 24] for each
#      individual station. The predictions are read from the results file for each split of the walk-forward validation
#      method. The predictions from each split are appended into one long list of predictions. The targets are pulled from
#      targets file in the GWN results directory. The MSE, MAE, RMSE, and SMAPE metrics are then calculated and written
#      to the metric files.

#      Parameters:
#          stations - List of the weather stations.
#          args - Parser of parameter arguments.
#      """
#     num_splits = 27
#     num_stations = 21
#     for station in range(num_stations):

#         for horizon in [3, 6, 9, 12, 24]:

#             pred = []
#             real = []

#             for split in range(num_splits):
#                 resultsFile = 'Results/GWN/' + str(horizon) + ' Hour Forecast/Predictions/outputs_' + str(split) + \
#                               '.pkl'
#                 targetsFile = 'Results/GWN/' + str(horizon) + ' Hour Forecast/Targets/targets_' + str(split) + '.pkl'
#                 yhat = utils.load_pickle(resultsFile)
#                 target = utils.load_pickle(targetsFile)
#                 pred.extend(np.array(yhat).flatten())
#                 real.extend(np.array(target).flatten())

#             pred = np.array(pred).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
#                                            args.seq_length))
#             real = np.array(real).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
#                                            args.seq_length))

#             metricFile = 'Results/GWN/Metrics/' + stations[station] + '/metrics_' + str(horizon)
#             file = open(metricFile, 'w')

#             preds = pred[:, station, :]
#             real_values = real[:, station, :]

#             root = metrics.rmse(real_values, preds)
#             square = metrics.mse(real_values, preds)
#             abs = metrics.mae(real_values, preds)
#             ape = metrics.smape(real_values, preds)

#             print('RMSE: {0} for station {1} forecasting {2} hours ahead'.format(root, station+1, horizon))
#             print('MSE: {0} for station {1} forecasting {2} hours ahead'.format(square, station+1, horizon))
#             print('MAE: {0} for station {1} forecasting {2} hours ahead'.format(abs, station+1, horizon))
#             print('SMAPE: {0} for station {1} forecasting {2} hours ahead'.format(ape, station+1, horizon))
#             print(' ')

#             file.write('This is the MSE ' + str(square) + '\n')
#             file.write('This is the MAE ' + str(abs) + '\n')
#             file.write('This is the RMSE ' + str(root) + '\n')
#             file.write('This is the SMAPE ' + str(ape) + '\n')

#             file.close()





# def eval(stations, args):
#     """
#      Calculates the GWN model's performance on the test set across all forecasting horizons [3, 6, 9, 12, 24] for each
#      individual station. The predictions are read from the results file for each split of the walk-forward validation
#      method. The predictions from each split are appended into one long list of predictions. The targets are pulled from
#      targets file in the GWN results directory. The MSE, MAE, RMSE, and SMAPE metrics are then calculated and written
#      to the metric files.

#      Parameters:
#          stations - List of the weather stations.
#          args - Parser of parameter arguments.
#      """
#     num_splits = 27
#     num_stations = 21

#     for station in range(num_stations):
#         for horizon in [3, 6, 9, 12, 24]:
#             pred = []
#             real = []

#             for split in range(num_splits):
#                 results_file = f'Results/GWN/{horizon} Hour Forecast/Predictions/outputs_{split}.pkl'
#                 targets_file = f'Results/GWN/{horizon} Hour Forecast/Targets/targets_{split}.pkl'

#                 try:
#                     yhat = utils.load_pickle(results_file)
#                     target = utils.load_pickle(targets_file)
#                     pred.extend(np.array(yhat).flatten())
#                     real.extend(np.array(target).flatten())
#                 except IOError:
#                     print(f'Error: Unable to read data from {results_file} or {targets_file}')
#                     continue

#             try:
#                 pred = np.array(pred).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
#                                                args.seq_length))
#                 real = np.array(real).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
#                                                args.seq_length))

#                 metric_file = f'Results/GWN/Metrics/{stations[station]}/metrics_{horizon}'

#                 with open(metric_file, 'w') as file:
#                     preds = pred[:, station, :]
#                     real_values = real[:, station, :]

#                     root = metrics.rmse(real_values, preds)
#                     square = metrics.mse(real_values, preds)
#                     abs_val = metrics.mae(real_values, preds)
#                     ape = metrics.smape(real_values, preds)

#                     print('RMSE: {0} for station {1} forecasting {2} hours ahead'.format(root, station+1, horizon))
#                     print('MSE: {0} for station {1} forecasting {2} hours ahead'.format(square, station+1, horizon))
#                     print('MAE: {0} for station {1} forecasting {2} hours ahead'.format(abs_val, station+1, horizon))
#                     print('SMAPE: {0} for station {1} forecasting {2} hours ahead'.format(ape, station+1, horizon))
#                     print('')

#                     file.write('This is the MSE ' + str(square) + '\n')
#                     file.write('This is the MAE ' + str(abs_val) + '\n')
#                     file.write('This is the RMSE ' + str(root) + '\n')
#                     file.write('This is the SMAPE ' + str(ape) + '\n')
#             except IOError:
#                 print(f'Error: Unable to write metrics to {metric_file}')



def eval(stations, args):
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

                    
                    yhat = utils.load_pickle(results_file)
                    target = utils.load_pickle(targets_file)
                    pred.extend(np.array(yhat).flatten())
                    real.extend(np.array(target).flatten())
                
                     # Reshape pred and real arrays
                    pred = np.array(pred).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
                                                args.seq_length))
                    real = np.array(real).reshape((int(len(real) / (args.n_stations * args.seq_length)), args.n_stations,
                                                args.seq_length))

                    metric_file = f'Results/GWN/Metrics/{stations[station]}/metrics_{horizon}'

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
            except IOError:
                metric_file = f'Results/GWN/Metrics/{stations[station]}/metrics_{horizon}'
                #print(f'Error: Unable to write metrics to {metric_file}')
                print('Error! : Unable to read data or write metrics for station {} and forecast length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))