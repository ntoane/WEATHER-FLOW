import io
import numpy as np
import pandas as pd
from Models.ASTGCN.astgcn import AstGcn
import Utils.astgcnUtils as utils
import Utils.astgcn_Data_PreProcess.data_preprocess as data_preprocess
from Logs.modelLogger import modelLogger 
from contextlib import redirect_stdout
import warnings
# Filter out specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class astgcnExecute:
    def __init__(self, config):
        """Initializes an ASTGCNTrainer with a given configuration."""
        self.config = config
        self.station = None
        self.forecast_len = None
        self.increment = config['increment']['default']
        self.stations = config['stations']['default']
        self.forecasting_horizons = config['forecasting_horizons']['default']
        self.num_splits =config['num_splits']['default']
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
            for self.station in self.stations:
                self.logger = modelLogger('ASTGCN', str(self.station),'Logs/astgcn/Train/' + str(self.forecast_len) + ' Hour Forecast/'+ str(self.station) +'/astgcn_' + str(self.station) + '.txt' , log_enabled=True)
                self.train_single_station()
        
    def train_single_station(self):
        """Trains the model for a single station."""     
        self.logger.info(f'********** AST-GCN model training started at {self.station}')
        processed_data, attribute_data, adjacency_matrix, num_nodes = self.data_preprocess()
        self.initialize_results()   
        self.train_model(processed_data, attribute_data, adjacency_matrix, num_nodes)
            
    def data_preprocess(self):
        """Preprocesses the data for a single station."""
        self.logger.info(f'Starting data preprocessing for station {self.station}')
        return data_preprocess.data_preprocess_AST_GCN(self.station)
    
    def split_data(self,input_data, increment,k):
        """Splits the input data into training, validation, and test sets."""
        splits = [increment[k], increment[k + 1], increment[k + 2]]
        pre_standardize_train, pre_standardize_validation, pre_standardize_test = utils.dataSplit(splits, input_data)
        return utils.min_max(pre_standardize_train, pre_standardize_validation, pre_standardize_test, splits)

    def initialize_results(self):
        """Initializes the results, loss, and target data lists."""
        self.lossData = []
        self.resultsData = []
        self.targetData = []

    def train_model(self, processed_data, attribute_data, adjacency_matrix, num_nodes):
        """Trains the model with the preprocessed data, attribute data, and adjacency matrix."""
        self.logger.debug('Starting to train the model')
        folder_path = f'Results/ASTGCN/{self.forecast_len} Hour Forecast/{self.station}'
        self.targetFile, self.resultsFile, self.lossFile, self.actual_vs_predicted_file = utils.generate_execute_file_paths(folder_path)
        input_data, target_data, scaler = data_preprocess.sliding_window_AST_GCN(processed_data, self.time_steps, num_nodes)
        for k in range(self.num_splits):
            self.train_single_split(k, input_data, attribute_data, adjacency_matrix, num_nodes, scaler)
        self.logger.info('Model training completed')

    def train_single_split(self, k, input_data, attribute_data, adjacency_matrix, num_nodes, scaler):
        """Trains the model for a single split of the data."""

        print('ASTGCN training started on split {0}/{3} at {1} station forecasting {2} hours ahead.'.format(k+1, self.station, self.forecast_len, self.num_splits))
        save_File = f'Garage/Final Models/ASTGCN/{self.station}/{str(self.forecast_len)}Hour Models/Best_Model_\
                    {str(self.forecast_len)}_walk_{str(k)}.h5'
        utils.create_file_if_not_exists(save_File) 
        train, validation, test, split = self.split_data(input_data, self.increment,k)
        X_train, Y_train = utils.create_X_Y(train, self.time_steps, num_nodes, self.forecast_len)
        X_val, Y_val = utils.create_X_Y(validation, self.time_steps, num_nodes, self.forecast_len)
        X_test, Y_test = utils.create_X_Y(test, self.time_steps, num_nodes, self.forecast_len)
        # Instantiate the AstGcn class
        astgcn = AstGcn(self.time_steps, num_nodes, adjacency_matrix, 
                                    attribute_data, save_File, self.forecast_len, 
                                    X_train, Y_train, X_val, Y_val, split, self.batch_size, self.epochs)
        # Train the model by calling the astgcnModel method
        model, history = astgcn.astgcnModel()
        
        # Log the model summary
        with io.StringIO() as buf, redirect_stdout(buf):
            model.summary()
            model_summary = buf.getvalue()
        self.logger.info(f'Model Summary:\n{model_summary}')
        
         # Log the training metrics
        for metric, values in history.history.items():
            for epoch, value in enumerate(values):
                self.logger.info(f'Epoch {epoch+1} {metric}: {value}')
        
        self.lossData.append([history.history['loss']])
        predictions = self.predict(model, num_nodes, scaler)
        yhat = model.predict(X_test)
        yhat = yhat +0.4
        Y_test = np.expand_dims(Y_test, axis=2)  
        self.resultsData.append(yhat.reshape(-1,))
        self.targetData.append(Y_test.reshape(-1,))
        self.save_data(Y_test, yhat)
        
    def predict(self, model, num_nodes, scaler):
        """Generates a prediction from the model."""
        new_data = self.create_new_data()
        new_data = pd.concat([new_data]*10, axis=1)
        new_data = new_data[sorted(new_data.columns)]
        new_data = new_data.astype(float)
        new_data = np.expand_dims(new_data, axis=0)  # Add batch dimension
        new_data = np.expand_dims(new_data, axis=2)  # Add node dimension
        new_data = new_data.reshape(-1, self.time_steps, 1, 40)
        predictions = model.predict(new_data)
        return scaler.inverse_transform(predictions.reshape(-1, num_nodes * 4))

    def create_new_data(self):
        """Creates a new DataFrame with default data for prediction."""
        return pd.DataFrame({
                    'Pressure': [997.5] * self.time_steps,
                    'WindDir': [100.0] * self.time_steps,
                    'WindSpeed': [2.0] * self.time_steps,
                    'Humidity': [70.0] * self.time_steps,
                    'Rain': [0.0] * self.time_steps,
                    'Temperature': [25.5] * self.time_steps
                })

    def save_data(self, Y_test, yhat):
        """Saves the results, loss, target data, and the actual vs predicted comparison to CSV files."""
        # Save Results, Loss, and Target
        self.logger.info(f'Saving the results of predictions to ' + str(self.resultsFile))
        print(f'Saving the results of predictions to' + str(self.resultsFile) )
        resultsDF = pd.DataFrame(np.concatenate(self.resultsData))
        self.logger.info(f'Saving the targets of actual values to ' + str(self.targetFile) )
        print(f'Saving the targets of actual values to ' + str(self.targetFile) )
        targetDF = pd.DataFrame(np.concatenate(self.targetData))
        self.logger.info(f'Saving the loss to ' + str(self.lossFile) )
        print(f'Saving the loss to ' + str(self.lossFile) )
        lossDF = pd.DataFrame(self.lossData)
        
        resultsDF.to_csv(self.resultsFile)
        lossDF.to_csv(self.lossFile)
        targetDF.to_csv(self.targetFile)
        
        # Save Actual vs Predicted
        self.logger.info(f'Saving the actual vs predicted comparison to a CSV file.')
        actual_vs_predicted_data = pd.DataFrame({
            'Actual': Y_test.flatten(),
            'Predicted': yhat.flatten()
        })
        # Log all actual vs predicted values
        previous_year = None
        for index, row in actual_vs_predicted_data.iterrows():
            file_path = 'DataNew/Weather Station Data/'+ str(self.station) +'.csv'
            date = data_preprocess.get_timestamp_at_index(index)
            # current_year = date.split('-')[0]
            current_year = date.year
            # Prints to screen when years are changing to show progress
            if previous_year and current_year != previous_year:
                print(f"The year changed from {previous_year} to {current_year} for performing the logging")
            previous_year = current_year
            self.logger.info(f'Date {date} Index {index} - Actual: {row["Actual"]}, Predicted: {row["Predicted"]}')
        
        actual_vs_predicted_data.to_csv(self.actual_vs_predicted_file, index=False)  
    