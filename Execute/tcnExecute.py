import Models.TCN.tcnone as tcn_one
import Models.TCN.tcntwo as tcn_two
import Utils.tcnUtils as utils
import pandas as pd
from keras.models import load_model
from tcn import TCN
import tensorflow as tf
import os
from Logs.modelLogger import modelLogger
from Execute.modelExecute import modelExecute
from datetime import datetime,timedelta
tf.config.threading.set_intra_op_parallelism_threads(24)
tf.config.threading.set_inter_op_parallelism_threads(24)

class tcnExecute(modelExecute):
    def __init__(self, sharedConfig, tcnConfig):
        super().__init__('tcn', sharedConfig, tcnConfig)
        self.model_logger = None

    def execute(self):
        # physical_devices = tf.config.list_physical_devices('CPU') #CPU
        # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

        increment = self.sharedConfig['increment']['default']
        stations = self.sharedConfig['stations']['default']
        forecasting_horizons = self.sharedConfig['horizons']['default']
        
        for forecast_len in forecasting_horizons:
            configFile = open("Execute/Best Configurations/tcn_params.txt", "r")
            for station in stations:
                # Making sure folder is created if doesnt exist
                log_path = 'Logs/TCN/Train/' + str(forecast_len) + ' Hour Forecast/' + str(station) + '/'
                os.makedirs(log_path, exist_ok=True)
                log_file = log_path + 'tcn_' + str(station) + '.txt'
                self.model_logger = modelLogger('tcn', str(station), log_file, log_enabled=True)
                print('Forecasting at station ', station)
                self.model_logger.info('TCN model training started at ' + station)
                print('TCN model training started at ' + station)
                # pulling in weather station data
                weatherData = 'DataNew/Weather Station Data/' + station + '.csv'
                

                ts = utils.create_dataset(weatherData)
                # reading in the parameters from the text file
                params = configFile.readline()
                cfg = utils.stringtoCfgTCN(params)
                # dynamically set hpo settings set for tcn model
                layers = int(cfg[0])
                filters = int(cfg[1])
                lag_length = int(cfg[2])
                batch = int(cfg[3])
                dropout = float(cfg[4])
                activation = cfg[5]
               
                # default settings from config file for tcn model are set when the model is initialized
                # This setting changes for each of the forecast_len in the above list for the horizon, thus not in config file
                n_ahead_length = forecast_len
                num_splits = self.sharedConfig['n_split']['default']
            
                

                for k in range(num_splits):
                    lossDF, resultsDF, targetDF, targetFile, resultsFile, lossFile = self.create_dataframes_and_dirs(forecast_len, station, k)
                    print('TCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len, num_splits))
                    self.model_logger.info('TCN Model on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len,num_splits))
                    lossFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                           'loss.csv'
                    saveFile = 'Garage/Final Models/TCN/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
                            + str(n_ahead_length) + '_walk_' + str(k) + '.h5'

                    split = [increment[k], increment[k + 1], increment[k + 2]]
                    pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(split, ts)

                    # Scaling the data
                    # allDataPath= 'DataNew/Graph Neural Network Data/Graph Station Data/Graph.csv'
                    # allData = utils.create_dataset(allDataPath)
                    # allDataTrain= allData[0: increment[k + 1]*len(stations)]
                    # train, validation, test = utils.min_max(pre_standardize_train,
                    #                     pre_standardize_validation,
                    #                     pre_standardize_test, allDataTrain)
                    train, validation, test = utils.min_max(pre_standardize_train,
                                                            pre_standardize_validation,
                                                            pre_standardize_test)
                    # Defining input shape
                    n_ft = train.shape[1]
                    # Creating the X and Y for forecasting, validation set and training
                    X_train, Y_train = utils.create_X_Y(train, lag_length, n_ahead_length)
                    X_val, Y_val = utils.create_X_Y(validation, lag_length, n_ahead_length)
                    X_test, Y_test = utils.create_X_Y(test, lag_length, n_ahead_length)
                    # Setting optimizer with config
                    if (self.modelConfig['use_optimizer']['default']):
                        opt = self.modelConfig['optimizer']['default']
                    else:
                        opt = self.sharedConfig['optimizer']['default']
                    # Setting loss with config
                    lossF = ['MSE', 'MAE', 'sparse_categorical_crossentropy', 'categorical_crossentropy']
                    if (self.modelConfig['use_loss_function']['default']):
                        if (self.modelConfig['loss_function']['default'] in lossF):
                            loss_function = self.modelConfig['loss_function']['default']
                        else: loss_function = 'MSE'
                    else:    
                        if (self.sharedConfig['loss_function']['default'] in lossF):
                            loss_function = self.sharedConfig['loss_function']['default']
                        else: loss_function = 'MSE'
                    # Creating the tcn model for temperature prediction
                    if layers == 1:
                        tcn_model = tcn_one.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                    n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
                                                    epochs=self.modelConfig['epochs']['default'], batch_size=self.modelConfig['batch_size']['default'], 
                                                    act_func=activation, loss=loss_function,
                                                    learning_rate=self.modelConfig['lr']['default'], batch_norm=self.modelConfig['batch_norm']['default'], 
                                                    layer_norm=self.modelConfig['layer_norm']['default'],
                                                    weight_norm=self.modelConfig['weight_norm']['default'], kernel=self.modelConfig['kernels']['default'], filters=filters,
                                                    dilations=self.modelConfig['dilations']['default'], padding=self.modelConfig['padding']['default'], dropout=dropout,
                                                    patience=self.modelConfig['patience']['default'], save=saveFile, optimizer=opt)
                        # Training the model
                        model, history = tcn_model.temperature_model()
                        # validation and train loss to dataframe
                        lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])
                        self.model_logger.info("Loss: " + str(history.history['loss']) )
                        self.model_logger.info("Val Loss: " + str(history.history['val_loss']) )
                        
                        # load best model
                        model = load_model(saveFile, custom_objects={'TCN': TCN})
                        # Test the model and write to file
                        yhat = model.predict(X_test)
                        # predictions to dataframe
                        resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])
                        
                        # self.save_actual_vs_predicted(Y_test, yhat, station,forecast_len)

                    else:
                        tcn_model = tcn_two.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                    n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
                                                    epochs=self.modelConfig['epochs']['default'], batch_size=self.modelConfig['batch_size']['default'], 
                                                    act_func=activation, loss=loss_function,
                                                    learning_rate=self.modelConfig['lr']['default'], batch_norm=self.modelConfig['batch_norm']['default'], 
                                                    layer_norm=self.modelConfig['layer_norm']['default'],
                                                    weight_norm=self.modelConfig['weight_norm']['default'], kernel=self.modelConfig['kernels']['default'], filters=filters,
                                                    dilations=self.modelConfig['dilations']['default'], padding=self.modelConfig['padding']['default'], dropout=dropout,
                                                    patience=self.modelConfig['patience']['default'], save=saveFile, optimizer=self.sharedConfig['optimizer']['default'])

                        # Training the model
                        model, history = tcn_model.temperature_model()
                        # validation and train loss to dataframe
                        lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])
                        self.model_logger.info("Loss: " + str(history.history['loss']) )
                        self.model_logger.info("Val Loss: " + str(history.history['val_loss']) )
                        
                        # load best model
                        model = load_model(saveFile, custom_objects={'TCN': TCN})
                        # Test the model and write to file
                        yhat = model.predict(X_test)
                        # predictions to dataframe
                        resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])
                        
                        # self.save_actual_vs_predicted(Y_test, yhat, station,forecast_len)

                    self.model_logger.info('TCN training done on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len,num_splits))
                    # Targets to dataframe
                    targetDF = pd.concat([targetDF, pd.Series(Y_test.reshape(-1, ))])

                    resultsDF.to_csv(resultsFile)
                    lossDF.to_csv(lossFile)
                    targetDF.to_csv(targetFile)
                
                self.model_logger.info('TCN training finished at ' + station)  
                
                # act_vs_predDF.to_csv(actuals_vs_predicted)
    
            configFile.close()
        self.model_logger.info('TCN training finished at all stations set for training :)')
        
    def save_actual_vs_predicted(self, Y_test, yhat, station,forecast_len):
        actual_vs_predicted_data = pd.DataFrame({
            'Actual': Y_test.flatten(),
            'Predicted': yhat.flatten()
        })

        # def get_timestamp_at_index(csv_file_path, index_to_find):
        #     # Read only the 'DateT' column
        #     df = pd.read_csv(csv_file_path, usecols=['DateT'])#, error_bad_lines=False)
        #     # Retrieve the DateT value at the specified index
        #     timestamp = df.loc[index_to_find, 'DateT']
        #     return timestamp

        def get_timestamp_at_index(hours):
            # Create a datetime object
            date_string = "2010-01-01 00:00:00"
            formatted_date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
            # Number of hours to add
            hours_to_add = hours

            # Create a timedelta representing the number of hours to add
            time_delta = timedelta(hours=hours_to_add)

            # Add the timedelta to the original date
            new_date = formatted_date + time_delta
            return new_date
        
        actual_vs_predicted_file = f'Results/TCN/{forecast_len} Hour Forecast/{station}/Predictions/actual_vs_predicted.csv'
        # actual_vs_predicted_data.to_csv(actual_vs_predicted_file, index=True)
        
        # Log all actual vs predicted values
        self.model_logger.info(f'Saving the actual vs predicted comparison to a CSV file - file path DataNew/Weather Station Data/' + str(station) + '.csv')
        previous_year = None
        for index, row in actual_vs_predicted_data.iterrows():
            file_path = 'DataNew/Weather Station Data/' + str(station) + '.csv'
            date = get_timestamp_at_index(index)
            current_year = date.year
            # print(date)
            
            # Prints to screen when years are changing to show progress
            if previous_year and current_year != previous_year:
                print(f"The year changed from {previous_year} to {current_year} for performing the logging")
            previous_year = current_year
            self.model_logger.info(f'Date {date} Index {index} - Actual: {row["Actual"]}, Predicted: {row["Predicted"]}')
        
    def create_dataframes_and_dirs(self, forecast_len, station, k):
        lossDF = pd.DataFrame()
        resultsDF = pd.DataFrame()
        targetDF = pd.DataFrame()

        base_path = f'Results/TCN/{forecast_len} Hour Forecast/{station}'

        target_path = f'{base_path}/Targets/'
        os.makedirs(target_path, exist_ok=True)
        targetFile = f'{target_path}target_'+ str(k) +'.csv'

        result_path = f'{base_path}/Predictions/'
        os.makedirs(result_path, exist_ok=True)
        resultsFile = f'{result_path}result_'+ str(k)  +'.csv'

        loss_path = f'{base_path}/Predictions/'
        os.makedirs(loss_path, exist_ok=True)
        lossFile = f'{loss_path}loss_'+ str(k)  +'.csv'

        return lossDF, resultsDF, targetDF, targetFile, resultsFile, lossFile
        