from abc import ABC, abstractmethod
from Logs.modelLogger import modelLogger
import numpy as np

class modelHPO(ABC):
    def __init__(self,model_name, sharedConfig, modelConfig):
        self.sharedConfig = sharedConfig
        self.gwnConfig = modelConfig
        self.best_cfg = []
        log_file = f'Logs/{model_name}/HPO/{model_name}_all_stations.txt'
        self.model_logger = modelLogger(model_name, 'all', log_file, log_enabled=False)
        self.best_cfg = []
        self. best_mse = np.inf
        # self.num_splits = 2

    @abstractmethod
    def train_model(self, X_train, Y_train, X_val, Y_val, cfg, saveFile, n_ft):
        pass

    @abstractmethod
    def hpo(self):
        pass
