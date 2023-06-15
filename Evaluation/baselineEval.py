import pandas as pd
import Utils.metrics as metrics

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



def eval(stations, model):
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