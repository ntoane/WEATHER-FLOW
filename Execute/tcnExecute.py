import Models.TCN.tcnone as tcn_one
import Models.TCN.tcntwo as tcn_two
import Utils.tcnUtils as utils
import pandas as pd
from keras.models import load_model
from tcn import TCN
import tensorflow as tf
import os
from Logs.modelLogger import modelLogger
from modelExecute import modelExecute

class TcnExecute(modelExecute):
    
    def __init__(self, sharedConfig, tcnConfig):
        super().__init__('tcn', sharedConfig, tcnConfig)

    def execute(self):
        
        # physical_devices = tf.config.list_physical_devices('CPU') #CPU
        # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

        increment = self.sharedConfig['increment']['default']
        stations = self.sharedConfig['stations']['default']
        forecasting_horizons = self.sharedConfig['horizons']['default']
        # self.tcn_logger = modelLogger('tcn', 'all','Logs/TCN/Train/'+'tcn_all_stations.txt', log_enabled=False)  
        self.tcn_logger.info('tcnTrain : TCN training started at all stations set for training :)') 

        for forecast_len in forecasting_horizons:
            configFile = open("Execute/Best Configurations/tcn_params.txt", "r")
            # self.tcn_logger = modelLogger('tcn', 'all','Evaluation/Logs/TCN/tcn_logs.txt')
        
            for station in stations:
                # printing out which station we are forecasting
                # self.tcn_logger = modelLogger('tcn', '{1}', 'TCN training started on split {0}/47 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                #                                                                                          forecast_len))
                
                self.tcn_logger = modelLogger('tcn', str(station),'Logs/TCN/Train/' + str(forecast_len) + ' Hour Forecast/'+str(station) +'/'+'tcn_' + str(station) + '.txt' , log_enabled=False)
                print('Forecasting at station ', station)
                #print('Evaluation/Logs/TCN/' + str(forecast_len) + ' Hour Forecast/'+str(station) +'/'+'tcn_' + str(station) + '.txt')
                self.tcn_logger.info('tcnTrain : TCN model training started at ' + station)
                print('tcnTrain : TCN model training started at ' + station)

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
                
                # layers, filters, lag_length,batch, dropout, activation = read_config()
                
                # default settings from config file for tcn model are set when the model is initialized
                
                # This setting changes for each of the forecast_len in the above list for the horizon, thus not in config file
                n_ahead_length = forecast_len
                
                lossDF = pd.DataFrame()
                resultsDF = pd.DataFrame()
                targetDF = pd.DataFrame()

                targetFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' + \
                            'target.csv'
                # Specify the directory path
                target_path = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' 
                # Create the directory if it doesn't exist
                os.makedirs(target_path, exist_ok=True)
                resultsFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                            'result.csv'
                result_path = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/'
                # Create the directory if it doesn't exist
                os.makedirs(result_path, exist_ok=True)
                lossFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                        'loss.csv'
                loss_path = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/'
                # Create the directory if it doesn't exist
                os.makedirs(loss_path, exist_ok=True)

                num_splits = self.sharedConfig['n_split']['default']# was 27

                for k in range(num_splits):
                    print('TCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len, num_splits))
                    self.tcn_logger.info('tcnTrain :TCN Model on split {0}/47 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len))

                    # lossFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
                    #        'loss.csv'
                    
                    
                    saveFile = 'Garage/Final Models/TCN/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
                            + str(n_ahead_length) + '_walk_' + str(k) + '.h5'

                    split = [increment[k], increment[k + 1], increment[k + 2]]
                    pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(split, ts)

                    # Scaling the data
                    train, validation, test = utils.min_max(pre_standardize_train,
                                                            pre_standardize_validation,
                                                            pre_standardize_test)

                    # Defining input shape
                    n_ft = train.shape[1]
                    
                    # Creating the X and Y for forecasting
                    X_train, Y_train = utils.create_X_Y(train, lag_length, n_ahead_length)

                    # Creating the X and Y for validation set
                    X_val, Y_val = utils.create_X_Y(validation, lag_length, n_ahead_length)

                    # Get the X feature set for training
                    X_test, Y_test = utils.create_X_Y(test, lag_length, n_ahead_length)

                    if (self.tcnConfig['use_optimizer']['default']):
                        opt = self.tcnConfig['optimizer']['default']
                    else:
                        opt = self.sharedConfig['optimizer']['default']
                        
                    
                    lossF = ['MSE', 'MAE', 'sparse_categorical_crossentropy', 'categorical_crossentropy']
                    if (self.tcnConfig['use_loss_function']['default']):
                        if (self.tcnConfig['loss_function']['default'] in lossF):
                            loss_function = self.tcnConfig['loss_function']['default']
                        else: loss_function = 'MSE'
                    else:    
                        if (self.sharedConfig['loss_function']['default'] in lossF):
                            loss_function = self.sharedConfig['loss_function']['default']
                        else: loss_function = 'MSE'

                    # Creating the tcn model for temperature prediction
                    if layers == 1:
                        tcn_model = tcn_one.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                    n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
                                                    epochs=self.tcnConfig['epochs']['default'], batch_size=self.tcnConfig['batch_size']['default'], 
                                                    act_func=activation, loss=loss_function,
                                                    learning_rate=self.tcnConfig['lr']['default'], batch_norm=self.tcnConfig['batch_norm']['default'], 
                                                    layer_norm=self.tcnConfig['layer_norm']['default'],
                                                    weight_norm=self.tcnConfig['weight_norm']['default'], kernel=self.tcnConfig['kernels']['default'], filters=filters,
                                                    dilations=self.tcnConfig['dilations']['default'], padding=self.tcnConfig['padding']['default'], dropout=dropout,
                                                    patience=self.tcnConfig['patience']['default'], save=saveFile, optimizer=opt)

                        # Training the model
                        model, history = tcn_model.temperature_model()

                        # validation and train loss to dataframe
                        lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])

                        # load best model
                        model = load_model(saveFile, custom_objects={'TCN': TCN})
                        # Test the model and write to file
                        yhat = model.predict(X_test)
                        # predictions to dataframe
                        resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])

                    else:
                        tcn_model = tcn_two.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                    n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
                                                    epochs=self.tcnConfig['epochs']['default'], batch_size=self.tcnConfig['batch_size']['default'], 
                                                    act_func=activation, loss=loss_function,
                                                    learning_rate=self.tcnConfig['lr']['default'], batch_norm=self.tcnConfig['batch_norm']['default'], 
                                                    layer_norm=self.tcnConfig['layer_norm']['default'],
                                                    weight_norm=self.tcnConfig['weight_norm']['default'], kernel=self.tcnConfig['kernels']['default'], filters=filters,
                                                    dilations=self.tcnConfig['dilations']['default'], padding=self.tcnConfig['padding']['default'], dropout=dropout,
                                                    patience=self.tcnConfig['patience']['default'], save=saveFile, optimizer=self.sharedConfig['optimizer']['default'])

                        # Training the model
                        model, history = tcn_model.temperature_model()

                        # validation and train loss to dataframe
                        lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])

                        # load best model
                        model = load_model(saveFile, custom_objects={'TCN': TCN})
                        # Test the model and write to file
                        yhat = model.predict(X_test)
                        # predictions to dataframe
                        resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])

                    self.tcn_logger.info('tcnTrain : TCN training done on split {0}/47 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
                                                                                                        forecast_len))
                    # Targets to dataframe
                    targetDF = pd.concat([targetDF, pd.Series(Y_test.reshape(-1, ))])
                
                self.tcn_logger.info('tcnTrain : TCN training finished at ' + station)  
                    
                resultsDF.to_csv(resultsFile)
                lossDF.to_csv(lossFile)
                targetDF.to_csv(targetFile)
        
            configFile.close()
        
        self.tcn_logger.info('tcnTrain : TCN training finished at all stations set for training :)')
        

    def read_config():
            cfg = open("Execute/Best Configurations/tcn_params.txt", "r")
            params = cfg.readline()
            cfg.close()
            utils.stringtoCfgTCN(params)
            # dynamically set hpo settings set for tcn model
            layers = int(cfg[0])
            filters = int(cfg[1])
            lag_length = int(cfg[2])
            batch = int(cfg[3])
            dropout = float(cfg[4])
            activation = cfg[5]
            return layers, filters, lag_length, batch, dropout, activation
    
    
    
    
# def train(self.sharedConfig,self.tcnConfig):
    
#     # physical_devices = tf.config.list_physical_devices('CPU') #CPU
#     # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#     increment = self.sharedConfig['increment']['default']
#     stations = self.sharedConfig['stations']['default']
#     forecasting_horizons = self.sharedConfig['horizons']['default']
#     self.tcn_logger = modelLogger('tcn', 'all','Logs/TCN/Train/'+'tcn_all_stations.txt', log_enabled=False)  
#     self.tcn_logger.info('tcnTrain : TCN training started at all stations set for training :)') 

#     for forecast_len in forecasting_horizons:
#         configFile = open("Execute/Best Configurations/tcn_params.txt", "r")
#         # self.tcn_logger = modelLogger('tcn', 'all','Evaluation/Logs/TCN/tcn_logs.txt')
    
#         for station in stations:
#             # printing out which station we are forecasting
#             # self.tcn_logger = modelLogger('tcn', '{1}', 'TCN training started on split {0}/47 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
#             #                                                                                          forecast_len))
            
#             self.tcn_logger = modelLogger('tcn', str(station),'Logs/TCN/Train/' + str(forecast_len) + ' Hour Forecast/'+str(station) +'/'+'tcn_' + str(station) + '.txt' , log_enabled=False)
#             print('Forecasting at station ', station)
#             #print('Evaluation/Logs/TCN/' + str(forecast_len) + ' Hour Forecast/'+str(station) +'/'+'tcn_' + str(station) + '.txt')
#             self.tcn_logger.info('tcnTrain : TCN model training started at ' + station)
#             print('tcnTrain : TCN model training started at ' + station)

#             # pulling in weather station data
#             weatherData = 'DataNew/Weather Station Data/' + station + '.csv'
#             ts = utils.create_dataset(weatherData)

#             # reading in the parameters from the text file
#             params = configFile.readline()
#             cfg = utils.stringtoCfgTCN(params)

#             # dynamically set hpo settings set for tcn model
#             layers = int(cfg[0])
#             filters = int(cfg[1])
#             lag_length = int(cfg[2])
#             batch = int(cfg[3])
#             dropout = float(cfg[4])
#             activation = cfg[5]
            
#             # default settings from config file for tcn model are set when the model is initialized
            
#             # This setting changes for each of the forecast_len in the above list for the horizon, thus not in config file
#             n_ahead_length = forecast_len
             
#             lossDF = pd.DataFrame()
#             resultsDF = pd.DataFrame()
#             targetDF = pd.DataFrame()

#             targetFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' + \
#                          'target.csv'
#             # Specify the directory path
#             target_path = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Targets/' 
#             # Create the directory if it doesn't exist
#             os.makedirs(target_path, exist_ok=True)
#             resultsFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
#                           'result.csv'
#             result_path = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/'
#             # Create the directory if it doesn't exist
#             os.makedirs(result_path, exist_ok=True)
#             lossFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
#                        'loss.csv'
#             loss_path = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/'
#             # Create the directory if it doesn't exist
#             os.makedirs(loss_path, exist_ok=True)

#             num_splits = self.sharedConfig['n_split']['default']# was 27

#             for k in range(num_splits):
#                 print('TCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, station,
#                                                                                                      forecast_len, num_splits))
#                 self.tcn_logger.info('tcnTrain :TCN Model on split {0}/47 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
#                                                                                                      forecast_len))

#                 # lossFile = 'Results/TCN/' + str(forecast_len) + ' Hour Forecast/' + station + '/Predictions/' + \
#                 #        'loss.csv'
                
                
#                 saveFile = 'Garage/Final Models/TCN/' + station + '/' + str(forecast_len) + ' Hour Models/Best_Model_' \
#                            + str(n_ahead_length) + '_walk_' + str(k) + '.h5'

#                 split = [increment[k], increment[k + 1], increment[k + 2]]
#                 pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(split, ts)

#                 # Scaling the data
#                 train, validation, test = utils.min_max(pre_standardize_train,
#                                                         pre_standardize_validation,
#                                                         pre_standardize_test)

#                 # Defining input shape
#                 n_ft = train.shape[1]
                
#                 # Creating the X and Y for forecasting
#                 X_train, Y_train = utils.create_X_Y(train, lag_length, n_ahead_length)

#                 # Creating the X and Y for validation set
#                 X_val, Y_val = utils.create_X_Y(validation, lag_length, n_ahead_length)

#                 # Get the X feature set for training
#                 X_test, Y_test = utils.create_X_Y(test, lag_length, n_ahead_length)

#                 if (self.tcnConfig['use_optimizer']['default']):
#                     opt = self.tcnConfig['optimizer']['default']
#                 else:
#                     opt = self.sharedConfig['optimizer']['default']
                    
                
#                 lossF = ['MSE', 'MAE', 'sparse_categorical_crossentropy', 'categorical_crossentropy']
#                 if (self.tcnConfig['use_loss_function']['default']):
#                     if (self.tcnConfig['loss_function']['default'] in lossF):
#                         loss_function = self.tcnConfig['loss_function']['default']
#                     else: loss_function = 'MSE'
#                 else:    
#                     if (self.sharedConfig['loss_function']['default'] in lossF):
#                         loss_function = self.sharedConfig['loss_function']['default']
#                     else: loss_function = 'MSE'

#                 # Creating the tcn model for temperature prediction
#                 if layers == 1:
#                     tcn_model = tcn_one.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
#                                                    n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
#                                                    epochs=self.tcnConfig['epochs']['default'], batch_size=self.tcnConfig['batch_size']['default'], 
#                                                    act_func=activation, loss=loss_function,
#                                                    learning_rate=self.tcnConfig['lr']['default'], batch_norm=self.tcnConfig['batch_norm']['default'], 
#                                                    layer_norm=self.tcnConfig['layer_norm']['default'],
#                                                    weight_norm=self.tcnConfig['weight_norm']['default'], kernel=self.tcnConfig['kernels']['default'], filters=filters,
#                                                    dilations=self.tcnConfig['dilations']['default'], padding=self.tcnConfig['padding']['default'], dropout=dropout,
#                                                    patience=self.tcnConfig['patience']['default'], save=saveFile, optimizer=opt)

#                     # Training the model
#                     model, history = tcn_model.temperature_model()

#                     # validation and train loss to dataframe
#                     lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])

#                     # load best model
#                     model = load_model(saveFile, custom_objects={'TCN': TCN})
#                     # Test the model and write to file
#                     yhat = model.predict(X_test)
#                     # predictions to dataframe
#                     resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])

#                 else:
#                     tcn_model = tcn_two.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
#                                                    n_lag=lag_length, n_features=n_ft, n_ahead=n_ahead_length,
#                                                    epochs=self.tcnConfig['epochs']['default'], batch_size=self.tcnConfig['batch_size']['default'], 
#                                                    act_func=activation, loss=loss_function,
#                                                    learning_rate=self.tcnConfig['lr']['default'], batch_norm=self.tcnConfig['batch_norm']['default'], 
#                                                    layer_norm=self.tcnConfig['layer_norm']['default'],
#                                                    weight_norm=self.tcnConfig['weight_norm']['default'], kernel=self.tcnConfig['kernels']['default'], filters=filters,
#                                                    dilations=self.tcnConfig['dilations']['default'], padding=self.tcnConfig['padding']['default'], dropout=dropout,
#                                                    patience=self.tcnConfig['patience']['default'], save=saveFile, optimizer=self.sharedConfig['optimizer']['default'])

#                     # Training the model
#                     model, history = tcn_model.temperature_model()

#                     # validation and train loss to dataframe
#                     lossDF = lossDF.append([[history.history['loss'], history.history['val_loss']]])

#                     # load best model
#                     model = load_model(saveFile, custom_objects={'TCN': TCN})
#                     # Test the model and write to file
#                     yhat = model.predict(X_test)
#                     # predictions to dataframe
#                     resultsDF = pd.concat([resultsDF, pd.Series(yhat.reshape(-1, ))])

#                 self.tcn_logger.info('tcnTrain : TCN training done on split {0}/47 at {1} station forecasting {2} hours ahead.'.format(k+1, station,
#                                                                                                      forecast_len))
#                 # Targets to dataframe
#                 targetDF = pd.concat([targetDF, pd.Series(Y_test.reshape(-1, ))])
              
#             self.tcn_logger.info('tcnTrain : TCN training finished at ' + station)  
                
#             resultsDF.to_csv(resultsFile)
#             lossDF.to_csv(lossFile)
#             targetDF.to_csv(targetFile)
    
#         configFile.close()
     
#     self.tcn_logger.info('tcnTrain : TCN training finished at all stations set for training :)')