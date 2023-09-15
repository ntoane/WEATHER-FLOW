import numpy as np
from tcn import TCN
import Utils.metrics as metrics
import Utils.tcnUtils as utils
import Models.TCN.tcnone as tcn_one
import Models.TCN.tcntwo as tcn_two
from keras.models import load_model
from Logs.modelLogger import modelLogger
from HPO.modelHPO import modelHPO

class TCNHPO(modelHPO):
    def __init__(self, sharedConfig, tcnConfig):
        super().__init__('tcn', sharedConfig, tcnConfig)

    def train_model(self, X_train, Y_train, X_val, Y_val, cfg, saveFile,n_ft):
        if cfg['Layers'] == 1:
            #tcn_model = tcn_one.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val, **cfg, optimizer=self.sharedConfig['optimizer']['default'])
            tcn_model = tcn_one.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                   n_lag=cfg['Lag'], n_features=n_ft, n_ahead=cfg['Forecast Horizon'],
                                                   epochs=cfg['Epochs'], batch_size=cfg['Batch Size'],
                                                   act_func=cfg['Activation'], loss=cfg['Loss'], learning_rate=cfg['lr'],
                                                   batch_norm=cfg['Batch Norm'], layer_norm=cfg['Layer Norm'],
                                                   weight_norm=cfg['Weight Norm'], kernel=cfg['Kernels'],
                                                   filters=cfg['Filters'], dilations=cfg['Dilations'],
                                                   padding=cfg['Padding'], dropout=cfg['Dropout'],
                                                   patience=cfg['Patience'], save=saveFile,  optimizer=self.sharedConfig['optimizer']['default'])

        else:
            # tcn_model = tcn_two.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val, **cfg, optimizer=self.sharedConfig['optimizer']['default'])
            tcn_model = tcn_two.temporalcn(x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val,
                                                   n_lag=cfg['Lag'], n_features=n_ft, n_ahead=cfg['Forecast Horizon'],
                                                   epochs=cfg['Epochs'], batch_size=cfg['Batch Size'],
                                                   act_func=cfg['Activation'], loss=cfg['Loss'], learning_rate=cfg['lr'],
                                                   batch_norm=cfg['Batch Norm'], layer_norm=cfg['Layer Norm'],
                                                   weight_norm=cfg['Weight Norm'], kernel=cfg['Kernels'],
                                                   filters=cfg['Filters'], dilations=cfg['Dilations'],
                                                   padding=cfg['Padding'], dropout=cfg['Dropout'],
                                                   patience=cfg['Patience'], save=saveFile,  optimizer=self.sharedConfig['optimizer']['default'])

        model, history = tcn_model.temperature_model()
        model = load_model(saveFile, custom_objects={'TCN': TCN})
        yhat = model.predict(X_val)
        return np.array(yhat.reshape(-1, ))

    def evaluate_config(self, cfg, ts, num_splits, increment, station):
        resultsArray = np.array([])
        targetArray = np.array([])
        for k in range(num_splits):
            saveFile = 'Garage/HPO Models/TCN/' + station + '/Best_Model_24' + '_walk_' + str(k) + '.h5'
            split = [increment[k], increment[k + 1], increment[k + 2]]
            pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(split, ts)
            train, validation, test = utils.min_max(pre_standardize_train, pre_standardize_validation, pre_standardize_test)
            n_ft = train.shape[1]
            X_train, Y_train = utils.create_X_Y(train, cfg['Lag'], cfg['Forecast Horizon'])
            X_val, Y_val = utils.create_X_Y(validation, cfg['Lag'], cfg['Forecast Horizon'])
            results = self.train_model(X_train, Y_train, X_val, Y_val, cfg, saveFile,n_ft)
            resultsArray = np.concatenate((resultsArray, results))
            targetArray = np.concatenate((targetArray, np.array(Y_val.reshape(-1, ))))
        ave_mse = metrics.mse(targetArray, resultsArray)
        return ave_mse, n_ft

    def hpo(self):
        increment = self.sharedConfig['increment']['default']
        stations = self.sharedConfig['stations']['default']
        num_splits = 2
        for station in stations:
            # self.model_logger = modelLogger('tcn', str(station),'Logs/TCN/HPO/' + str(station) +'/'+'tcn_' + str(station) + '.txt', log_enabled=False)
            print('Performing TCN random search HPO at station: ', station)
            self.model_logger.info('tcnHPO : TCN HPO training started at ' + station)
            weatherData = 'DataNew/Weather Station Data/' + station + '.csv'
            ts = utils.create_dataset(weatherData)
            textFile = 'HPO/Best Parameters/TCN/' + station + '_configurations.txt'
            f = open(textFile, 'w')
            for i in range(self.sharedConfig['num_configs']['default']):
                cfg = utils.generateRandomTCNParameters()
                mse, n_ft = self.evaluate_config(cfg, ts, num_splits, increment, station)
                f.write('Configuration parameters at station ' + station + ': ' + str(cfg) + ' with MSE =' + str(mse) + '\n')
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_cfg = cfg
            self.model_logger.info('tcnHPO : Best parameters found at station ' + station + ': ' + str(self.best_cfg) + ' with MSE =' + str(self.best_mse) + '\n')
            f.write('Best parameters found at station ' + station + ': ' + str(self.best_cfg) + ' with MSE =' + str(self.best_mse) + '\n')
            f.close()


