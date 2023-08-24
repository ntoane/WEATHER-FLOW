import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tcn import TCN
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.threading.set_intra_op_parallelism_threads(24)
tf.config.threading.set_inter_op_parallelism_threads(24)

class temporalcn:
    def __init__(self, x_train, y_train, x_val, y_val, n_lag, n_features, n_ahead, epochs, batch_size,
                 act_func, loss, learning_rate, batch_norm, layer_norm, weight_norm, kernel,
                 filters, dilations, padding, dropout, patience, save, optimizer):
        """"
        Initialize a temporalcn object.

        Args:
            x_train (ndarray): The input training data.
            y_train (ndarray): The target training data.
            x_val (ndarray): The input validation data.
            y_val (ndarray): The target validation data.
            n_lag (int): The number of lagged time steps to consider as input.
            n_features (int): The number of features in the input data.
            n_ahead (int): The number of time steps to predict.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size for training.
            act_func (str): The activation function to use in the TCN layers.
            loss (str): The loss function to use for training.
            learning_rate (float): The learning rate for the optimizer.
            batch_norm (bool): Whether to use batch normalization in the TCN layers.
            layer_norm (bool): Whether to use layer normalization in the TCN layers.
            weight_norm (bool): Whether to use weight normalization in the TCN layers.
            kernel (int): The kernel size for the TCN layers.
            filters (int): The number of filters in the TCN layers.
            dilations (list): The dilation factors for the TCN layers.
            padding (str): The padding mode for the TCN layers.
            dropout (float): The dropout rate for the TCN layers.
            patience (int): The number of epochs to wait for improvement in validation loss before early stopping.
            save (str): The path to save the best model checkpoint.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.n_lag = n_lag
        self.n_features = n_features
        self.n_ahead = n_ahead
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_func = act_func
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.weight_norm = weight_norm
        self.kernel = kernel
        self.filters = filters
        self.dilations = dilations
        self.padding = padding
        self.dropout = dropout
        self.patience = patience
        self.save = save
        self.optimizer = optimizer
        

    def temperature_model(self):
        """
        Build and train the temperature model.

        Returns:
            model (Sequential): The trained temperature model.
            history (History): The training history.
        """
        
        # Build the model architecture
        model = Sequential()
        model.add(TCN(
            input_shape=(self.n_lag, self.n_features),
            activation=self.act_func,
            nb_filters=self.filters,
            kernel_size=self.kernel,
            dilations=self.dilations,
            dropout_rate=self.dropout,
            use_batch_norm=self.batch_norm,
            use_weight_norm=self.weight_norm,
            use_layer_norm=self.layer_norm,
            return_sequences=False,
            padding=self.padding
        ))
        model.add(Dense(self.n_ahead, activation="linear"))

        # Configure optimizer and compile the model
        if (self.optimizer == 'SGD'):
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, decay=1e-2 / self.epochs)
        elif (self.optimizer == 'RMSprop'):
            opt = keras.optimizers.RMSprop(learning_rate=self.learning_rate, decay=1e-2 / self.epochs)
        else: opt = keras.optimizers.Adam(learning_rate=self.learning_rate, decay=1e-2 / self.epochs)
        model.compile(loss=self.loss,
                      optimizer=opt,
                      metrics=[self.loss, 'mape'])

        # Define callbacks for early stopping and model checkpointing
        early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)
        checkpoint = ModelCheckpoint(self.save, save_weights_only=False, monitor='val_loss', verbose=1,
                                     save_best_only=True,
                                     mode='min', save_freq='epoch')
        callback = [early_stop, checkpoint]

        # Train the model
        history = model.fit(self.x_train, self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=callback)

        return model, history
