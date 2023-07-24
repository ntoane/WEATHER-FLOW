from abc import ABC, abstractmethod
from Logs.modelLogger import modelLogger

class modelExecute(ABC):
    def __init__(self, model_name, sharedConfig, tcnConfig):
        self.sharedConfig = sharedConfig
        self.modelConfig = tcnConfig
        log_file = f'Logs/{model_name}/Train/{model_name}_all_stations.txt'
        self.model_logger = modelLogger(model_name, 'all', log_file, log_enabled=False) 

    @abstractmethod
    def execute(self):
        pass

    