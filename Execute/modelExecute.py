from abc import ABC, abstractmethod
from Logs.modelLogger import modelLogger

class modelExecute(ABC):
    def __init__(self, model_name, sharedConfig, tcnConfig):
        self.sharedConfig = sharedConfig
        self.modelConfig = tcnConfig
        

    @abstractmethod
    def execute(self):
        pass

    