import pandas as pd
import Utils.gwnUtils as util
import torch
import time
from Engine.gwnEngine import trainer
import numpy as np
import Utils.metrics as metrics
import warnings
warnings.filterwarnings("error")
from Logs.modelLogger import modelLogger
import yaml

def train_model(config, data, split, supports, adj_init, model_file):
    """
    Trains a GWN model and calculates MSE on validation set for random search HPO. Train and validation sets scaled
    using MinMax normalization. Scaled train and validation data then processed into sliding window input-output pairs.
    Scaled sliding-window data then fed into DataLoaders. GWN model then trained on training data and tested on
    validation data.

    Parameters:
        data_split - Split of data within walk-forward validation.
        config - Configuration file of parameters.
        train_data - Training data used to train GWN model.
        validate_data - Validation data on which the GWN model is tested.
        lag - length of the input sequence
        forecast - forecasting horizon(length of output), set to 24 hours.
        model_file - File of the best model on the validation set.

    Returns:
        predictions - Returns predictions made by the GWN model on the validation set.
        targets - Returns the target set.
    """

    train_data = data[0]
    validate_data = data[1]

    scaler = util.NormScaler(train_data.min(), train_data.max())

    engine = trainer(scaler, supports, adj_init, config)

    x_train, y_train = util.sliding_window(scaler.transform(train_data), config['lag_length']['default'], config['seq_length']['default'], split, 0, config['n_stations']['default'])
    x_validation, y_validation = util.sliding_window(scaler.transform(validate_data), config['lag_length']['default'], config['seq_length']['default'],
                                                     split, 1, config['n_stations']['default'])

    trainLoader = util.DataLoader(x_train, y_train, config['batch_size']['default'])
    validationLoader = util.DataLoader(x_validation, y_validation, config['batch_size']['default'])

    min_val_loss = np.inf
    trainLossArray = []
    validationLossArray = []

    for epoch in range(config['epochs']['default']):
        patience = 0
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
            '{:5.4f} '.format(epoch + 1, trainTime, train_loss, validationTime, validation_loss))

        if min_val_loss > validation_loss:
            min_val_loss = validation_loss
            patience = 0
            util.save_model(engine.model, model_file)
        else:
            patience += 1

        if patience == config['patience']['default']:
            break

    engine.model = util.load_model(model_file)
    testStart = time.time()
    validation_test_loss, predictions, targets = engine.test(validationLoader, config)
    testTime = time.time() - testStart

    print('Inference Time: {:4.2f}s | Loss on validation set: {:5.4f} '.format(
        testTime, validation_test_loss))

    return predictions, targets


def hpo(initialConfig):
    """
    Performs random search HPO on the GWN model. Trains a group of GWN models with different hyper-parameters on a train
    set and then tests the models' performance on the validation set. The configuration with the lowest MSE is then
    written to a file.
    Parameters:
        config -  Configuration file of parameters.
        increment - Walk-forward validation split points.
    """
    
    #dont need this as config is passed?
    # Load the YAML config file
    with open('config.yaml', 'r') as file:
        initialConfig = yaml.safe_load(file)
    
    increment = initialConfig['increment']['default']

    data = pd.read_csv(initialConfig['data']['default'])
    data = data.drop(['StasName', 'DateT','Latitude', 'Longitude'], axis=1)  #added latitude and longitude
    
    # gwn_logger = modelLogger('gwn','all' 'Evaluation/Logs/GWN/gwn_logs.txt')
    # gwn_logger.info('gwnHPO : Locating the best configuration settings.')
    
    gwn_logger = modelLogger('gwn', 'all','Logs/GWN/HPO/'+'gwn_all_stations.txt', log_enabled=False)
    print('Performing GWN random search HPO at all stations: ')
    gwn_logger.info('tcnHPO : TCN HPO started at all stations :)')

    textFile = 'HPO/Best Parameters/GWN/configurations.txt'
    f = open(textFile, 'w')

    best_mse = np.inf
    best_cfg = []

    num_splits = 2
    for i in range(initialConfig['num_configs']['default']):
        config = util.generateRandomParameters(initialConfig)
        valid_config = True
        targets = []
        preds = []

        for k in range(num_splits):
            modelFile = 'Garage/HPO Models/GWN/model_split_' + str(k)
            
            split = [increment[k] * int(initialConfig['n_stations']['default']), 
                     increment[k + 1] * int(initialConfig['n_stations']['default']),
                     increment[k + 2] * initialConfig['n_stations']['default']]
            data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]
            adj_matrix = util.load_adj(adjFile=initialConfig['adjdata']['default'], 
                                       adjtype=initialConfig['adjtype']['default'])
            supports = [torch.tensor(i).to(initialConfig['device']['default']) for i in adj_matrix]

            if initialConfig['randomadj']['default']:
                adjinit = None
            else:
                adjinit = supports[0]
            if initialConfig['aptonly']['default']:
                supports = None

            torch.manual_seed(0)

            try:
                
                print('This is the HPO configuration: \n',
                      'Dropout - ', initialConfig['dropout']['default'], '\n',
                      'Lag_length - ', initialConfig['lag_length']['default'], '\n',
                      'Hidden Units - ', initialConfig['nhid']['default'], '\n',
                      'Layers - ', initialConfig['num_layers']['default'], '\n',
                      'Batch Size - ', initialConfig['batch_size']['default'], '\n',
                      'Epochs - ', initialConfig['epochs']['default'])

                output, real = train_model(initialConfig, data_sets, split, supports, adjinit, modelFile)

            except Warning:
                valid_config = False
                break

            targets.append(np.array(real).flatten())
            preds.append(np.array(output).flatten())

        if valid_config:
            mse = metrics.mse(np.concatenate(np.array(targets, dtype=object)),
                              np.concatenate(np.array(preds, dtype=object)))
            if mse < best_mse:
                best_cfg = config
                best_mse = mse

    f.write('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
    f.close()
    textFile.close()
    gwn_logger.info('gwnHPO : GWN best configuration found = ' +str(best_cfg) + ' with an MSE of ' + str(best_mse))
    gwn_logger.info('tcnHPO : TCN HPO finished at all stations :)')
