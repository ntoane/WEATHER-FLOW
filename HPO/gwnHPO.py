import pandas as pd
import Utils.gwnUtils as util
import Utils.sharedUtils as sharedUtil
import torch
import time
from Models.GWN.gwnEngine import trainer
import numpy as np
import Utils.metrics as metrics
import warnings
warnings.filterwarnings("error")
from Logs.modelLogger import modelLogger
from HPO.modelHPO import modelHPO

class GWNHPO(modelHPO):
    def __init__(self, sharedConfig, gwnConfig):
        super().__init__('gwn', sharedConfig, gwnConfig)

    def hpo(self):
        increment = self.sharedConfig['increment']['default']
        data = self.prepare_data() 
        textFile = 'HPO/Best Parameters/GWN/configurations.txt'
        f = open(textFile, 'w')

        # best_mse = np.inf

        num_splits = 2
        for i in range(self.sharedConfig['num_configs']['default']):
            config = util.generateRandomParameters(self.gwnConfig)
            valid_config = True
            targets = []
            preds = []

            for k in range(num_splits):
                modelFile = 'Garage/HPO Models/GWN/model_split_' + str(k)
                n_stations= int(self.sharedConfig['n_stations']['default'])
                data_sets, split = self.split_data(data, increment, k, n_stations)
                # split = [increment[k] * n_stations, increment[k + 1] * n_stations, increment[k + 2] * n_stations]
                # data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]
                
                adj_matrix = util.load_adj(adjFile=self.gwnConfig['adjdata']['default'], 
                                           adjtype=self.gwnConfig['adjtype']['default'])
                supports = [torch.tensor(i).to(self.gwnConfig['device']['default']) for i in adj_matrix]

                if self.gwnConfig['randomadj']['default']:
                    adjinit = None
                else:
                    adjinit = supports[0]
                if self.gwnConfig['aptonly']['default']:
                    supports = None

                torch.manual_seed(0)

                try:
                    print('This is the HPO configuration: \n',
                          'Dropout - ', self.gwnConfig['dropout']['default'], '\n',
                          'Lag_length - ', self.gwnConfig['lag_length']['default'], '\n',
                          'Hidden Units - ', self.gwnConfig['nhid']['default'], '\n',
                          'Layers - ', self.gwnConfig['num_layers']['default'], '\n',
                          'Batch Size - ', self.gwnConfig['batch_size']['default'], '\n',
                          'Epochs - ', self.gwnConfig['epochs']['default'])

                    output, real = self.train_model(self.sharedConfig, self.gwnConfig, data_sets, split, supports, adjinit, modelFile)
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
        self.model_logger.info('gwnHPO : GWN best configuration found = ' +str(best_cfg) + ' with an MSE of ' + str(best_mse))
        self.model_logger.info('gwnHPO : GWN HPO finished at all stations :)')
        
    def prepare_data(self):
        data = pd.read_csv(self.gwnConfig['data']['default'])
        data = data.drop(['StasName', 'DateT','Latitude', 'Longitude'], axis=1)  #added latitude and longitude
        return data

    def split_data(self, data, increment, k, n_stations):
        split = [increment[k] * n_stations, increment[k + 1] * n_stations, increment[k + 2] * n_stations]
        data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]
        return data_sets, split

    def train_model(self,sharedConfig, gwnConfig,  data, split, supports, adj_init, model_file):
        trainLoader, validationLoader, scaler = self.create_train_and_validation_data(data, split)
        engine = self.train_and_validate_model(scaler,trainLoader, validationLoader, supports, adj_init, model_file)
        return self.test_model(engine, validationLoader, model_file)
    
    def create_train_and_validation_data(self, data, split):
        train_data = data[0]
        validate_data = data[1]
        scaler = util.NormScaler(train_data.min(), train_data.max())

        x_train, y_train = sharedUtil.sliding_window(scaler.transform(train_data), 
                                                     self.gwnConfig['lag_length']['default'], 
                                                     self.gwnConfig['seq_length']['default'], 
                                                     split, 
                                                     0, 
                                                     self.sharedConfig['n_stations']['default'])
        x_validation, y_validation = sharedUtil.sliding_window(scaler.transform(validate_data), 
                                                               self.gwnConfig['lag_length']['default'], 
                                                               self.gwnConfig['seq_length']['default'],
                                                               split, 
                                                               1, 
                                                               self.sharedConfig['n_stations']['default'])
        trainLoader = util.DataLoader(x_train, y_train, self.gwnConfig['batch_size']['default'])
        validationLoader = util.DataLoader(x_validation, y_validation, self.gwnConfig['batch_size']['default'])

        return trainLoader, validationLoader, scaler

    def train_and_validate_model(self, scaler, trainLoader, validationLoader, supports, adj_init, model_file):
        engine = trainer(scaler, supports, adj_init, self.sharedConfig, self.gwnConfig)
        min_val_loss = np.inf

        for epoch in range(self.gwnConfig['epochs']['default']):
            patience = 0
            trainStart = time.time()
            train_loss = engine.train(trainLoader, self.gwnConfig)
            trainTime = time.time() - trainStart

            validationStart = time.time()
            validation_loss = engine.validate(validationLoader, self.gwnConfig)
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

            if patience == self.gwnConfig['patience']['default']:
                break

        return engine

    def test_model(self, engine, validationLoader, model_file):
        engine.model = util.load_model(model_file)
        testStart = time.time()
        validation_test_loss, predictions, targets = engine.test(validationLoader, self.gwnConfig)
        testTime = time.time() - testStart

        print('Inference Time: {:4.2f}s | Loss on validation set: {:5.4f} '.format(
            testTime, validation_test_loss))

        return predictions, targets

