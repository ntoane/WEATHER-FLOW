import pandas as pd
import Utils.gwnUtils as util
import Utils.sharedUtils as sharedUtil
import torch
import time
from Models.GWN.gwnEngine import trainer
import numpy as np
import pickle
from Logs.modelLogger import modelLogger
from modelExecute import modelExecute

class GwnExecute(modelExecute):
    def __init__(self, sharedConfig, gwnConfig):
       super().__init__('gwn', sharedConfig, gwnConfig)
    
    def execute(self):
        increment = self.sharedConfig['increment']['default']
        data = self.prepare_data()
        forecast_horizons = self.sharedConfig['horizons']['default']

        for forecast_len in forecast_horizons:
            self.gwnConfig['seq_length']['default'] = forecast_len
            print('Training GWN models through walk-forward validation on a forecasting horizon of: ', self.gwnConfig['seq_length']['default'])
            self.gwn_logger.info('gwnTrain : Training GWN models through walk-forward validation on a forecasting horizon of: '+ str(self.gwnConfig['seq_length']['default']))
            
            for k in range(self.sharedConfig['n_split']['default']):
                fileDictionary = self.prepare_file_dictionary(forecast_len, k)
                split = self.prepare_data_split(increment, k)
                data_sets = self.split_data(data, split)
                supports, adjinit = self.prepare_supports_and_adjinit()
                torch.manual_seed(0)
                self.gwn_logger.info('gwnTrain : GWN model initialised.')

                train_data = data_sets[0]
                validate_data = data_sets[1]
                test_data = data_sets[2]
                trainLoader, validationLoader, testLoader, scaler = self.get_data_loaders(train_data, validate_data, test_data, split)
                engine = trainer(scaler, supports, adjinit, self.sharedConfig, self.gwnConfig)

                trainLossArray, validationLossArray = self.train_and_validate(engine, trainLoader, validationLoader, fileDictionary)
                test_loss, predictions, targets = self.test_model(engine, testLoader)
                self.save_results(fileDictionary, predictions, targets, trainLossArray, validationLossArray, engine)

                self.gwn_logger.info('gwnTrain : GWN model training done.')

    def prepare_data(self):
        data = pd.read_csv(self.gwnConfig['data']['default'])
        data = data.drop(['StasName', 'DateT', 'Latitude', 'Longitude'], axis=1)
        return data
    
    def split_data(self, data, split):
        return [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]

    def get_data_loaders(self, train_data, validate_data, test_data, split):
        scaler = util.NormScaler(train_data.min(), train_data.max())
        x_train, y_train = sharedUtil.sliding_window(scaler.transform(train_data), self.gwnConfig['lag_length']['default'], self.gwnConfig['seq_length']['default'], split, 0, self.sharedConfig['n_stations']['default'])
        x_validation, y_validation = sharedUtil.sliding_window(scaler.transform(validate_data), self.gwnConfig['lag_length']['default'], self.gwnConfig['seq_length']['default'], split, 1, self.sharedConfig['n_stations']['default'])
        x_test, y_test = sharedUtil.sliding_window(scaler.transform(test_data), self.gwnConfig['lag_length']['default'], self.gwnConfig['seq_length']['default'], split, 2, self.sharedConfig['n_stations']['default'])

        trainLoader = util.DataLoader(x_train, y_train, self.gwnConfig['batch_size']['default'])
        validationLoader = util.DataLoader(x_validation, y_validation, self.gwnConfig['batch_size']['default'])
        testLoader = util.DataLoader(x_test, y_test, self.gwnConfig['batch_size']['default'])

        return trainLoader, validationLoader, testLoader, scaler

    def train_and_validate(self, engine, trainLoader, validationLoader, dictionary):
        min_val_loss = np.inf
        trainLossArray = []
        validationLossArray = []
        for epoch in range(self.gwnConfig['epochs']['default']):
            trainStart = time.time()
            train_loss = engine.train(trainLoader, self.gwnConfig)
            trainLossArray.append(train_loss)
            trainTime = time.time() - trainStart

            validationStart = time.time()
            validation_loss = engine.validate(validationLoader, self.gwnConfig)
            validationLossArray.append(validation_loss)
            validationTime = time.time() - validationStart

            print('Epoch {:2d} | Train Time: {:4.2f}s | Train Loss: {:5.4f} | Validation Time: {:5.4f} | Validation Loss: {:5.4f} '.format(
                epoch + 1, trainTime, train_loss, validationTime, validation_loss))

            if min_val_loss > validation_loss:
                min_val_loss = validation_loss
                patience = 0
                util.save_model(engine.model, dictionary['modelFile'])
            else:
                patience += 1

            if patience == self.gwnConfig['patience']['default']:
                break

        engine.model = util.load_model(dictionary['modelFile'])
        return trainLossArray, validationLossArray

    def test_model(self, engine, testLoader):
        testStart = time.time()
        test_loss, predictions, targets = engine.test(testLoader, self.gwnConfig)
        testTime = time.time() - testStart

        print('Inference Time: {:4.2f}s | Test Loss: {:5.4f} '.format(testTime, test_loss))

        return test_loss, predictions, targets

    def save_results(self, dictionary, predictions, targets, trainLossArray, validationLossArray, engine):
        output = open(dictionary['predFile'], 'wb')
        pickle.dump(predictions, output)
        output.close()

        target = open(dictionary['targetFile'], 'wb')
        pickle.dump(targets, target)
        target.close()

        trainLossFrame = pd.DataFrame(trainLossArray)
        trainLossFrame.to_csv(dictionary['trainLossFile'])
        validationLossFrame = pd.DataFrame(validationLossArray)
        validationLossFrame.to_csv(dictionary['validationLossFile'])
        adjDataFrame = util.get_adj_matrix(engine.model)
        adjDataFrame.to_csv(dictionary['matrixFile'])

    def prepare_file_dictionary(self, forecast_len, k):
        fileDictionary = {
            'predFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Predictions/outputs_' + str(k) + '.pkl',
            'targetFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Targets/targets_' + str(k) + '.pkl',
            'trainLossFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
            'validationLossFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
            'modelFile': 'Garage/Final Models/GWN/' + str(forecast_len) + ' Hour Models/model_split_' + str(k),
            'matrixFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv'
        }
        return fileDictionary

    def prepare_data_split(self, increment, k):
        split = [increment[k] * self.sharedConfig['n_stations']['default'],
                 increment[k + 1] * self.sharedConfig['n_stations']['default'],
                 increment[k + 2] * self.sharedConfig['n_stations']['default']]
        return split

    def prepare_supports_and_adjinit(self):
        adj_matrix = util.load_adj(adjFile=self.gwnConfig['adjdata']['default'], adjtype=self.gwnConfig['adjtype']['default'])
        supports = [torch.tensor(i).to(self.gwnConfig['device']['default']) for i in adj_matrix]

        if self.gwnConfig['randomadj']['default']:
            adjinit = None
        else:
            adjinit = supports[0]

        if self.gwnConfig['aptonly']['default']:
            supports = None
            adjinit = None

        return supports, adjinit








# Old way of executing Gwn -> less modular
# def train_model(sharedConfig, gwnConfig, data_sets, split, supports, adj_init, dictionary):
#     train_data = data_sets[0]
#     validate_data = data_sets[1]
#     test_data = data_sets[2]

#     scaler = util.NormScaler(train_data.min(), train_data.max())

#     engine = trainer(scaler, supports, adj_init, sharedConfig, gwnConfig)

#     x_train, y_train = sharedUtil.sliding_window(scaler.transform(train_data), gwnConfig['lag_length']['default'], gwnConfig['seq_length']['default'], split, 0, sharedConfig['n_stations']['default'])
#     x_validation, y_validation = sharedUtil.sliding_window(scaler.transform(validate_data), gwnConfig['lag_length']['default'], gwnConfig['seq_length']['default'],
#                                                      split, 1, sharedConfig['n_stations']['default'])
#     x_test, y_test = sharedUtil.sliding_window(scaler.transform(test_data), gwnConfig['lag_length']['default'], gwnConfig['seq_length']['default'], split, 2, sharedConfig['n_stations']['default'])

#     trainLoader = util.DataLoader(x_train, y_train, gwnConfig['batch_size']['default'])
#     validationLoader = util.DataLoader(x_validation, y_validation, gwnConfig['batch_size']['default'])
#     testLoader = util.DataLoader(x_test, y_test, gwnConfig['batch_size']['default'])

#     min_val_loss = np.inf
#     trainLossArray = []
#     validationLossArray = []
#     for epoch in range(gwnConfig['epochs']['default']):
#         trainStart = time.time()
#         train_loss = engine.train(trainLoader, gwnConfig)
#         trainLossArray.append(train_loss)
#         trainTime = time.time() - trainStart

#         validationStart = time.time()
#         validation_loss = engine.validate(validationLoader, gwnConfig)
#         validationLossArray.append(validation_loss)
#         validationTime = time.time() - validationStart
        

#         print(
#             'Epoch {:2d} | Train Time: {:4.2f}s | Train Loss: {:5.4f} | Validation Time: {:5.4f} | Validation Loss: '
#             '{:5.4f} '.format(
#                 epoch + 1, trainTime, train_loss, validationTime, validation_loss))

#         if min_val_loss > validation_loss:
#             min_val_loss = validation_loss
#             patience = 0
#             util.save_model(engine.model, dictionary['modelFile'])

#         else:
#             patience += 1

#         if patience == gwnConfig['patience']['default']:
#             break

# #         print(util.get_adj_matrix(engine.model))
#     engine.model = util.load_model(dictionary['modelFile'])
#     #gwn_logger.info('GWN model initialized.')

#     testStart = time.time()
#     test_loss, predictions, targets = engine.test(testLoader, gwnConfig)
#     testTime = time.time() - testStart

#     print('Inference Time: {:4.2f}s | Test Loss: {:5.4f} '.format(
#         testTime, test_loss))

#     output = open(dictionary['predFile'], 'wb')
#     pickle.dump(predictions, output)
#     output.close()

#     target = open(dictionary['targetFile'], 'wb')
#     pickle.dump(targets, target)
#     target.close()

#     trainLossFrame = pd.DataFrame(trainLossArray)
#     trainLossFrame.to_csv(dictionary['trainLossFile'])
#     validationLossFrame = pd.DataFrame(validationLossArray)
#     validationLossFrame.to_csv(dictionary['validationLossFile'])
#     adjDataFrame = util.get_adj_matrix(engine.model)
#     adjDataFrame.to_csv(dictionary['matrixFile'])


# def train(sharedConfig, gwnConfig):
   
#     increment = sharedConfig['increment']['default']
    
#     gwn_logger = modelLogger('gwn', 'all', 'Logs/GWN/Train/gwn_all_stations.txt', log_enabled=False) 
    
#     # data is the weather station ?
#     # no it is Data/Graph Neural Network Data/Graph Station Data/Graph.csv 
#     # this is a csv file with all weather stations in 1 file, each hour by hour
#     data = pd.read_csv(gwnConfig['data']['default'])
#     #print(args.data)
#     data = data.drop(['StasName', 'DateT', 'Latitude', 'Longitude'], axis=1)  #added latitude and longitude

#     forecast_horizons = sharedConfig['horizons']['default']

#     for forecast_len in forecast_horizons:
#         gwnConfig['seq_length']['default'] = forecast_len
#         #print('Training WGN models through walk-forward validation on a forecasting horizon of: ', config['seq_length']['default'])
#         print('Training GWN models through walk-forward validation on a forecasting horizon of: ', gwnConfig['seq_length']['default'])
#         gwn_logger.info('gwnTrain : Training GWN models through walk-forward validation on a forecasting horizon of: '+ str(gwnConfig['seq_length']['default']))
#         for k in range(sharedConfig['n_split']['default']):
#             fileDictionary = {'predFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Predictions/outputs_' +
#                                           str(k) + '.pkl',
#                               'targetFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Targets/' + 'targets_'
#                                             + str(k) + '.pkl',
#                               'trainLossFile': 'Results/GWN/' + str(forecast_len) +
#                                                ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
#                               'validationLossFile': 'Results/GWN/' + str(forecast_len) +
#                                                     ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
#                               'modelFile': 'Garage/Final Models/GWN/' + str(forecast_len) +
#                                            ' Hour Models/model_split_' + str(k),
#                               'matrixFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Matrices/adjacency_matrix_' +
#                                            str(k) + '.csv'
#             }

#             split = [increment[k] * sharedConfig['n_stations']['default'], increment[k + 1] * sharedConfig['n_stations']['default'], increment[k + 2] *
#                      sharedConfig['n_stations']['default']]
#             data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]

#             adj_matrix = util.load_adj(adjFile=gwnConfig['adjdata']['default'], adjtype=gwnConfig['adjtype']['default'])
#             supports = [torch.tensor(i).to(gwnConfig['device']['default']) for i in adj_matrix]

#             if gwnConfig['randomadj']['default']:
#                 adjinit = None
#             else:
#                 adjinit = supports[0]
#             if gwnConfig['aptonly']['default']:
#                 supports = None
#                 adjinit = None

#             torch.manual_seed(0)
#             gwn_logger.info('gwnTrain : GWN model initialised.')
#             train_model(sharedConfig, gwnConfig, data_sets, split, supports, adjinit, fileDictionary)
#             gwn_logger.info('gwnTrain : GWN model training done.')