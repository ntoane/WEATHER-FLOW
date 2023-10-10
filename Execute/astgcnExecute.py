import io
import os
import numpy as np
import pandas as pd
from Models.ASTGCN.astgcn import AstGcn
import Utils.astgcnUtils as utils
import Utils.astgcn_Data_PreProcess.data_preprocess as data_preprocess
from Logs.modelLogger import modelLogger 
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class astgcnExecute:
    def __init__(self, sharedConfig, config):
        """Initializes an ASTGCNTrainer with a given configuration."""
        self.config = config
        self.station = None
        self.forecast_len = None
        self.increment = sharedConfig['increment']['default']
        self.stations = sharedConfig['stations']['default']
        self.forecasting_horizons = sharedConfig['horizons']['default']
        self.num_splits =sharedConfig['n_split']['default']
        self.time_steps =config['time_steps']['default']
        self.batch_size = config['batch_size']['default']
        self.epochs = config['training_epoch']['default']
        self.logger = None

    def train(self):
        """Trains the model for all forecast lengths and stations. Either set to single or multiple 
        time steps to forecast"""
        print("Executing experimentation for time series prediction for weather forecasting")
        print("Forecasting horizons currently set to " + str(self.forecasting_horizons));
        for self.forecast_len in self.forecasting_horizons:
            log_dir = 'Logs/ASTGCN/Train/' + str(self.forecast_len) + ' Hour Forecast/'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.logger = modelLogger('ASTGCN', "All_Stations", log_dir +  "_All_Stations.txt", log_enabled=True)
            self.train_single_station()
             
    def train_single_station(self):
        """Trains the model for a single station."""     
        self.logger.info(f'********** AST-GCN model training started')
        processed_data, attribute_data, adjacency_matrix, num_nodes = data_preprocess.data_preprocess_AST_GCN()
        self.initialize_results()   
        self.train_model(processed_data, attribute_data, adjacency_matrix, num_nodes)
            
    def split_data(self,input_data, increment,k):
        """Splits the input data into training, validation, and test sets."""
        splits = [increment[k], increment[k + 1], increment[k + 2]]
        standardized_train, standardized_validation, standardized_test = utils.dataSplit(splits, input_data)
        return (standardized_train, standardized_validation, standardized_test, splits)

    def initialize_results(self):
        """Initializes the results, loss, and target data lists."""
        self.lossData = []
        self.resultsData = []
        self.targetData = []

    def train_model(self, processed_data, attribute_data, adjacency_matrix, num_nodes):
        """Trains the model with the preprocessed data, attribute data, and adjacency matrix."""
        self.logger.debug('Starting to train the model')
        folder_path = f'Results/ASTGCN/{self.forecast_len} Hour Forecast/All Stations'
        self.targetFile, self.resultsFile, self.lossFile, self.actual_vs_predicted_file = utils.generate_execute_file_paths(folder_path)
        input_data, target_data = data_preprocess.sliding_window_AST_GCN(processed_data, self.time_steps, num_nodes)
        for k in range(self.num_splits):
            self.train_single_split(k, input_data, attribute_data, adjacency_matrix, num_nodes)
        self.logger.info('Model training completed')

    def train_single_split(self, k, input_data, attribute_data, adjacency_matrix, num_nodes):
        """Trains the model for a single split of the data."""
        print('ASTGCN training started on split {0}/{2} at all stations forecasting {1} hours ahead.'.format(k+1, self.forecast_len, self.num_splits))
        save_File = f'Garage/Final Models/ASTGCN/All Stations/{str(self.forecast_len)}Hour Models/Best_Model_\
                    {str(self.forecast_len)}_walk_{str(k)}.h5'
        utils.create_file_if_not_exists(save_File) 
        
        ### Normalize input data as a whole
        def normalize_data(data):
            min_val = np.min(data)
            max_val = np.max(data)
            normalized_data = (data - min_val) / (max_val - min_val)
            return normalized_data
        
        # Normalizing splits of data
        def normalize_splits(pre_standardize_train, pre_standardize_validation, pre_standardize_test, splits):
            min_val = np.min(pre_standardize_train)
            max_val = np.max(pre_standardize_train)
            
            # Normalizing the data
            train_data = (pre_standardize_train - min_val) / (max_val - min_val)
            val_data = (pre_standardize_validation - min_val) / (max_val - min_val)
            test_data = (pre_standardize_test - min_val) / (max_val - min_val)
            
            return train_data, val_data, test_data, splits
        
        # Normalize the input data then split it into train,test,validate
        input_data = normalize_data(input_data)
        train, validation, test, split = self.split_data(input_data, self.increment,k)
        
        X_train, Y_train = utils.create_X_Y(train, self.time_steps, num_nodes, self.forecast_len)
        X_val, Y_val = utils.create_X_Y(validation, self.time_steps, num_nodes, self.forecast_len)
        X_test, Y_test = utils.create_X_Y(test, self.time_steps, num_nodes, self.forecast_len)
        
        ## normalize the attribute data too
        attribute_data = normalize_data(attribute_data)
        
        # Instantiate the AstGcn class
        astgcn = AstGcn(self.time_steps, num_nodes, adjacency_matrix, 
                                    attribute_data, save_File, self.forecast_len, 
                                    X_train, Y_train, X_val, Y_val, split, self.batch_size, self.epochs, 
                                    self.config['gru_units']['default'], self.config['lstm_neurons']['default'])
        # Train the model by calling the astgcnModel method
        model, history = astgcn.astgcnModel()
        
        self.lossData.append([history.history['loss']])
        yhat = model.predict(X_test)

        self.resultsData.append(yhat.reshape(-1,))
        self.targetData.append(Y_test.reshape(-1,))
        self.save_data(Y_test, yhat)

    def save_data(self, Y_train, yhat):
        """Saves the results, loss, target data, and the actual vs predicted comparison to CSV files."""
        # Save Results, Loss, and Targe
        resultsDF = pd.DataFrame(np.concatenate(self.resultsData))
        targetDF = pd.DataFrame(np.concatenate(self.targetData))
        lossDF = pd.DataFrame(self.lossData)
        resultsDF.to_csv(self.resultsFile)
        lossDF.to_csv(self.lossFile)
        targetDF.to_csv(self.targetFile)
        
        # Save Actual vs Predicted
        actual_vs_predicted_data = pd.DataFrame({
            'Actual': Y_train.flatten(),
            'Predicted': yhat.flatten()
        })
        for index, row in actual_vs_predicted_data.iterrows():
            file_path = 'data/Weather Station Data/'+ str(self.station) +'.csv'
            date = data_preprocess.get_timestamp_at_index(index)
            self.logger.info(f'Date {date} Index {index} - Actual: {row["Actual"]}, Predicted: {row["Predicted"]}')
            # print(f'Date {date} Index {index} - Actual: {row["Actual"]}, Predicted: {row["Predicted"]}')