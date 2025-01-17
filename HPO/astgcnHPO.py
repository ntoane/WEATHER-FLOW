import numpy as np
import pandas as pd
import os
from Models.ASTGCN.astgcn import AstGcn
import Utils.astgcnUtils as utils
from Utils.astgcn_Data_PreProcess.data_preprocess import data_preprocess_AST_GCN, sliding_window_AST_GCN
from Logs.modelLogger import modelLogger 

class astgcnHPO:
    def __init__(self, sharedConfig, config):
        self.config = config
        
    def hpo(self):
        increment = self.config['increment']['default']
        stations = self.config['stations']['default']
        num_splits = self.config['num_splits']['default']
        time_steps = self.config['time_steps']['default']
        batch_size = self.config['batch_size']['default']
        epochs = self.config['training_epoch']['default']
        horizon  = 24
        
        param_path = 'HPO/Best Parameters/ASTGCN/'
        if not os.path.exists(param_path):
            os.makedirs(param_path)
        f = open(param_path + "configurations.txt", 'w')
        log_path = 'Logs/ASTGCN/HPO/'
        os.makedirs(log_path, exist_ok=True)
        log_file = log_path + 'astgcn_all_stations.txt'
        self.model_logger = modelLogger('astgcn', 'all_stations', log_file, log_enabled=True)
       
        print('********** AST-GCN model HPO started at all stations') 
        
        textFile = 'HPO/Best Parameters/AST-GCN/configurations.txt'
        if not os.path.exists('HPO/Best Parameters/AST-GCN/'):
            os.makedirs('HPO/Best Parameters/AST-GCN/')
        print("File with best configs is in ", textFile)
        
        with open(textFile, 'w') as f:
            best_smape = np.inf
                
            processed_data, attribute_data, adjacency_matrix, num_nodes = data_preprocess_AST_GCN()
            lossData, resultsData, targetData = [], [], []
                    
            folder_path = f'Results/ASTGCN/HPO/{horizon} Hour Forecast/all_stations'
            targetFile, resultsFile, lossFile, actual_vs_predicted_file = utils.generate_execute_file_paths(folder_path)
            input_data, target_data = sliding_window_AST_GCN(processed_data, time_steps, num_nodes)

            num_splits = 1
            for i in range(self.config['num_configs']['default']):
                config = utils.generateRandomParameters(self.config)
                print(f"Trying configuration {i+1}/{self.config['num_configs']['default']}: {config}")
                self.model_logger.info("Generating random parameters for ASTGCN")
                self.model_logger.info(f"Trying configuration {i+1}/{self.config['num_configs']['default']}: {config}")
                        
                valid_config = True
                targets = []
                preds = []
                    
                for k in range(num_splits):
                    print('ASTGCN HPO training started on split {0}/{2} at all stations forecasting {1} hours ahead.'.format(k + 1, horizon, num_splits))
                    
                    save_File = f'Garage/Final Models/ASTGCN/All Stations/{str(horizon)}Hour Models/Best_Model_\
                        {str(horizon)}_walk_{str(k)}.h5'
                    utils.create_file_if_not_exists(save_File) 
                    
                    # Normalize the input data then split it into train,test,validate
                    input_data = utils.normalize_data(input_data)
                    train, validation, test, split = utils.split_data(input_data, increment,k)
                        
                    X_train, Y_train = utils.create_X_Y(train, time_steps, num_nodes, horizon)
                    X_val, Y_val = utils.create_X_Y(validation, time_steps, num_nodes, horizon)
                    X_test, Y_test = utils.create_X_Y(test, time_steps, num_nodes, horizon)
                    try:
                        print('This is the HPO configuration: \n',
                            'Batch Size - ', self.config['batch_size']['default'], '\n',
                            'Epochs - ', self.config['training_epoch']['default'], '\n',
                            'Hidden GRU units - ', self.config['gru_units']['default'], '\n'
                            'LSTM units - ', self.config['lstm_neurons']['default'], '\n'
                            ) 
                        attribute_data = utils.normalize_data(attribute_data)
                        
                        # Instantiation and training
                        astgcn = AstGcn(time_steps, num_nodes, adjacency_matrix,
                                                        attribute_data, save_File, horizon,
                                                        X_train, Y_train, X_val, Y_val, split, 
                                                        self.config['batch_size']['default'], self.config['training_epoch']['default'], 
                                                        self.config['gru_units']['default'], self.config['lstm_neurons']['default'])
                        model, history = astgcn.astgcnModel()
                        lossData.append([history.history['loss']])
                        yhat = model.predict(X_test)
                        Y_test = np.expand_dims(Y_test, axis=2)
                        resultsData.append(yhat.reshape(-1,))
                        targetData.append(Y_test.reshape(-1,))
                    except Warning:
                        valid_config = False
                        print(f"Error encountered during training with configuration {config}. Error message: {e}")
                        break
                    targets.append(np.array(targetData).flatten())
                    preds.append(np.array(resultsData).flatten())
                if valid_config:
                        
                    smape = utils.SMAPE(np.concatenate(np.array(targets, dtype=object)),
                                    np.concatenate(np.array(preds, dtype=object)))
                    if smape < best_smape:  # Note that "less is better" for SMAPE
                        print(f"Current smape {smape:.2f} is better than previous best smape {best_smape:.2f}.")
                        self.model_logger.info(f"Current smape {smape:.2f} is better than previous best smape {best_smape:.2f}.")
                        best_cfg = config
                        best_smape = smape
                    else:
                        print(f"Current smape {smape:.2f} is NOT better than previous best smape {best_smape:.2f}.")
                        self.model_logger.info(f"Current smape {smape:.2f} is NOT better than previous best smape {best_smape:.2f}.")

            # f.write('This is the best configuration ' + str(best_cfg) + ' with an smape of ' + str(best_smape))
        with open(textFile, 'a') as f:
            f.write('This is the best configuration ' + str(best_cfg) + ' with an smape of ' + str(best_smape))

        print('This is the best configuration ' + str(best_cfg) + ' with an smape of ' + str(best_smape))
        f.close()
        self.model_logger.info('This is the best configuration ' + str(best_cfg) + ' with an smape of ' + str(best_smape))
        self.model_logger.info("HPO finished successfully")
        print(f'HPO finished at all stations at {horizon} hour horizon')