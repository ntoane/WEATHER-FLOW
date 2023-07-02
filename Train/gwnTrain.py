import pandas as pd
import Utils.gwnUtils as util
import torch
import time
from Engine.gwnEngine import trainer
import numpy as np
import pickle
import logging
from Logs.modelLogger import modelLogger
import yaml


def train_model(config, data_sets, split, supports, adj_init, dictionary):
    """
    Trains a GWN model and calculates MSE on test set. Train, validation, and test  sets scaled using
    MinMax normalization. Scaled train, validation, and test data then processed into sliding window input-output pairs.
    Scaled sliding-window data then fed into DataLoaders. GWN model then trained on training data and tested on
    test data.

    Parameters:
        data_split - Split of data within walk-forward validation.
        config - Configuration file with parameters.
        train_data - Training data used to train GWN model.
        validate_data - Validation data on which the GWN model is validated.
        test_data - Test data on which the GWN model is tested.
        lag - Length of the input sequence.
        forecast - forecasting horizon(length of output), set to 24 hours.
        adj_init - Adjacency Matrix initialisation
        model_file - File of the best model on the validation set.
        pred_file - File to write predictions of GWN model to.
        target_file - File to write targets of test set to.
        train_loss_file - File to write loss(MSE) on train set to.
        validation_loss_file - File to write loss(MSE) on validation set to.
        matrix_file - File to write adjacency matrix to.
    """

    train_data = data_sets[0]
    validate_data = data_sets[1]
    test_data = data_sets[2]

    scaler = util.NormScaler(train_data.min(), train_data.max())

    engine = trainer(scaler, supports, adj_init, config)

    x_train, y_train = util.sliding_window(scaler.transform(train_data), config['lag_length']['default'], config['seq_length']['default'], split, 0, config['n_stations']['default'])
    x_validation, y_validation = util.sliding_window(scaler.transform(validate_data), config['lag_length']['default'], config['seq_length']['default'],
                                                     split, 1, config['n_stations']['default'])
    x_test, y_test = util.sliding_window(scaler.transform(test_data), config['lag_length']['default'], config['seq_length']['default'], split, 2, config['n_stations']['default'])

    trainLoader = util.DataLoader(x_train, y_train, config['batch_size']['default'])
    validationLoader = util.DataLoader(x_validation, y_validation, config['batch_size']['default'])
    testLoader = util.DataLoader(x_test, y_test, config['batch_size']['default'])

    min_val_loss = np.inf
    trainLossArray = []
    validationLossArray = []
    for epoch in range(config['epochs']['default']):
        trainStart = time.time()
        train_loss = engine.train(trainLoader, config)
        trainLossArray.append(train_loss)
        trainTime = time.time() - trainStart

        validationStart = time.time()
        validation_loss = engine.validate(validationLoader, config)
        validationLossArray.append(validation_loss)
        validationTime = time.time() - validationStart
        

        print(
            'Epoch {:2d} | Train Time: {:4.2f}s | Train Loss: {:5.4f} | Validation Time: {:5.4f} | Validation Loss: '
            '{:5.4f} '.format(
                epoch + 1, trainTime, train_loss, validationTime, validation_loss))

        if min_val_loss > validation_loss:
            min_val_loss = validation_loss
            patience = 0
            util.save_model(engine.model, dictionary['modelFile'])

        else:
            patience += 1

        if patience == config['patience']['default']:
            break

#         print(util.get_adj_matrix(engine.model))
    engine.model = util.load_model(dictionary['modelFile'])
    #gwn_logger.info('GWN model initialized.')

    testStart = time.time()
    test_loss, predictions, targets = engine.test(testLoader, config)
    testTime = time.time() - testStart

    print('Inference Time: {:4.2f}s | Test Loss: {:5.4f} '.format(
        testTime, test_loss))

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


def train(config):
    """
    Trains and tests the final GWN model through walk-forward validation across all 5 foreacasting horizons.
    A GWN model, with the same parameters, is trained across 27 splits. The predictions, targets, losses and adjacency
    matrices for each split are written to files for evaluation.

    Parameters:
        Config -  Default parameters from the configuration file.
        increment - Walk-forward validation split points.
    """

    #dont need this as config is passed?
    # Load the YAML config file
    # with open('config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    increment = config['increment']['default']
    
    gwn_logger = modelLogger('gwn', 'all', 'Logs/GWN/Train/gwn_all_stations.txt', log_enabled=False) 
    
    # data is the weather station ?
    # no it is Data/Graph Neural Network Data/Graph Station Data/Graph.csv 
    # this is a csv file with all weather stations in 1 file, each hour by hour
    data = pd.read_csv(config['data']['default'])
    #print(args.data)
    data = data.drop(['StasName', 'DateT', 'Latitude', 'Longitude'], axis=1)  #added latitude and longitude

    forecast_horizons = config['horizons']['default']

    for forecast_len in forecast_horizons:
        config['seq_length']['default'] = forecast_len
        #print('Training WGN models through walk-forward validation on a forecasting horizon of: ', config['seq_length']['default'])
        print('Training GWN models through walk-forward validation on a forecasting horizon of: ', config['seq_length']['default'])
        gwn_logger.info('gwnTrain : Training GWN models through walk-forward validation on a forecasting horizon of: '+ str(config['seq_length']['default']))
        for k in range(config['n_split']['default']):
            fileDictionary = {'predFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Predictions/outputs_' +
                                          str(k) + '.pkl',
                              'targetFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Targets/' + 'targets_'
                                            + str(k) + '.pkl',
                              'trainLossFile': 'Results/GWN/' + str(forecast_len) +
                                               ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                              'validationLossFile': 'Results/GWN/' + str(forecast_len) +
                                                    ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
                              'modelFile': 'Garage/Final Models/GWN/' + str(forecast_len) +
                                           ' Hour Models/model_split_' + str(k),
                              'matrixFile': 'Results/GWN/' + str(forecast_len) + ' Hour Forecast/Matrices/adjacency_matrix_' +
                                           str(k) + '.csv'
            }

            split = [increment[k] * config['n_stations']['default'], increment[k + 1] * config['n_stations']['default'], increment[k + 2] *
                     config['n_stations']['default']]
            data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]

            adj_matrix = util.load_adj(adjFile=config['adjdata']['default'], adjtype=config['adjtype']['default'])
            supports = [torch.tensor(i).to(config['device']['default']) for i in adj_matrix]

            if config['randomadj']['default']:
                adjinit = None
            else:
                adjinit = supports[0]
            if config['aptonly']['default']:
                supports = None
                adjinit = None

            torch.manual_seed(0)
            gwn_logger.info('gwnTrain : GWN model initialised.')
            train_model(config, data_sets, split, supports, adjinit, fileDictionary)
            gwn_logger.info('gwnTrain : GWN model training done.')
            
