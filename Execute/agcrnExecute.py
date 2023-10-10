import torch
import math
import os
import time
import copy
import numpy as np
import pandas as pd
from Models.AGCRN.AGCRN import AGCRN as Network
import torch.nn as nn
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
import torch
import numpy as np
import torch.nn as nn
from Models.AGCRN.AGCRN import AGCRN as Network
import Utils.agcrnUtils as agcrnUtil
import Utils.sharedUtils as sharedUtil
import Utils.gwnUtils as gwnUtil
from Execute.modelExecute import modelExecute
from Logs.modelLogger import modelLogger 
import csv
torch.set_num_threads(4)


class agcrnExecute(modelExecute):
    
    def __init__(self, sharedConfig, modelConfig):
        super().__init__('agcrn', sharedConfig, modelConfig)

    def execute(self):
        forecast_horizons = self.sharedConfig['horizons']['default']
        #iterate for all horizons
        for forecast_len in forecast_horizons:
            self.modelConfig['horizon']['default']=forecast_len
            #iterate for all splits
            for split in range(self.sharedConfig['n_split']['default']):
                #set up logging
                log_path = 'Logs/AGCRN/Train/' + str(forecast_len) + ' Hour Forecast/'
                os.makedirs(log_path, exist_ok=True)
                log_file = log_path + 'agcrn_all_stations.txt'
                self.model_logger = modelLogger('agcrn', 'all_stations', log_file, log_enabled=True)
                
                #logs info
                self.model_logger.info("Training AGCRN for horizons:"+ str(forecast_len) + "   split:"+str(split))
                print("Training AGCRN for horizons:"+ str(forecast_len) + "   split:"+str(split))

                #prep data and init model according to horizon and split
                self.prepare_file_dictionary(forecast_len, split)
                self.prep_split_data(forecast_len, split)
                self.initialise_model()
                self.dictionaryFile=self.prepare_file_dictionary
                self.execute_split(self.fileDictionary)         #execute the current split


    def initialise_model(self):
        DEVICE = 'cuda:0'
        self.modelConfig['device']['default']=DEVICE
        agcrnUtil.init_seed(self.modelConfig['seed']['default'])

        if torch.cuda.is_available():
            torch.cuda.set_device()
        else:
            DEVICE= 'cpu'

        #init model
        model = Network(self.modelConfig)
        model = model.to(DEVICE)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        if self.modelConfig['loss_func']['default'] == 'mae':
            loss = torch.nn.L1Loss().to(DEVICE)
        elif self.modelConfig['loss_func']['default'] == 'mse':
            loss = torch.nn.MSELoss().to(DEVICE)
        else:
            raise ValueError

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.modelConfig['lr_init']['default'], eps=1.0e-8,
                                    weight_decay=0, amsgrad=False)
        #learning rate decay
        lr_scheduler = None
        if self.modelConfig['lr_decay']['default']:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(self.modelConfig['lr_decay_step']['default'].split(','))]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                                milestones=lr_decay_steps,
                                                                gamma=self.modelConfig['lr_decay_rate']['defaults'])

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    


    def execute_split(self, fileDictionary):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()

        for epoch in range(1, self.modelConfig['epochs']['default'] + 1):
            train_epoch_loss = self.train_epoch(epoch)
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.model_logger.warning('Gradient explosion detected. Ending...')
                print('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.modelConfig['early_stop']['default']:
                if not_improved_count == self.modelConfig['early_stop_patience']['default']:
                    self.model_logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.modelConfig['early_stop_patience']['default']))
                    print("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.modelConfig['early_stop_patience']['default']))
                    break
            # save the best state
            if best_state == True:
                self.model_logger.info('*********************************Current best model saved!')
                print('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.model_logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        #save the best model to file
        if not self.modelConfig['debug']['default']:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            parent_dir = os.path.dirname(current_dir)
            path = os.path.join(parent_dir, fileDictionary['modelFile'])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(best_model, path)
            self.save_gateMatrix(self.model.gateMatrix, fileDictionary)
            self.save_updateMatrix(self.model.updateMatrix, fileDictionary)
            
            self.model_logger.info("Saving current best model to " + path) #self.best_path
            print("Saving current best model to " + path) #self.best_path

        #test
        self.model.load_state_dict(best_model)        
        self.test(self.model, self.modelConfig, self.test_loader, self.scaler,self.model_logger)


    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):

                data = data[..., :self.modelConfig['input_dim']['default']]  
                label = target[..., :self.modelConfig['output_dim']['default']]   
                output = self.model(data, target, teacher_forcing_ratio=0.)
                loss = self.loss(output.cpu(), label) #changes from cuda
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)

        self.model_logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.modelConfig['input_dim']['default']]  #input data
            label = target[..., :self.modelConfig['output_dim']['default']]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.modelConfig['teacher_forcing']['default']:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.modelConfig['tf_decay_steps']['default'])
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = self.loss(output.cpu(), label) #was cpu
            loss.backward()

            # add max grad clipping
            if self.modelConfig['grad_norm']['default']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.modelConfig['max_grad_norm']['default'])
            self.optimizer.step()
            total_loss += loss.item()
      
            #log information
            if batch_idx % self.modelConfig['log_step']['default'] == 0:
                # print("reach A2")
                self.model_logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
                print('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
   
        train_epoch_loss = total_loss/self.train_per_epoch
        self.model_logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))
        print('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))

        #learning rate decay
        if self.modelConfig['lr_decay']['default']:
            self.lr_scheduler.step()
        return train_epoch_loss



    def prep_split_data(self, horizon, k):
        increments = self.sharedConfig['increment']['default']
        n_stations = self.sharedConfig['n_stations']['default']
        self.train_loader, self.val_loader, self.test_loader, self.scaler = agcrnUtil.get_dataloader(horizon, k, 
                                                                    increments, n_stations, self.modelConfig,
                                                                    normalizer=self.modelConfig['normalizer']['default'],
                                                                    tod=self.modelConfig['tod']['default'], dow=False,
                                                                    weather=False, single=False)


        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)


    def prepare_file_dictionary(self, forecast_len, k):
        self.fileDictionary = {
            'predFile': './Results/AGCRN/' + str(forecast_len) + ' Hour Forecast/Predictions/outputs_' + str(k),
            'targetFile': 'Results/AGCRN/' + str(forecast_len) + ' Hour Forecast/Targets/targets_' + str(k),
            'trainLossFile': 'Results/AGCRN/' + str(forecast_len) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
            'validationLossFile': 'Results/AGCRN/' + str(forecast_len) + ' Hour Forecast/Matrices/adjacency_matrix_' + str(k) + '.csv',
            'modelFile': 'Garage/Final Models/AGCRN/' + str(forecast_len) + ' Hour Models/model_split_' + str(k) + ".pth",
            'updateMatrixFile': 'Results/AGCRN/' + str(forecast_len) + ' Hour Forecast/Matrices_update/adjacency_matrix_' + str(k) + '.csv',
            'gateMatrixFile': 'Results/AGCRN/' + str(forecast_len) + ' Hour Forecast/Matrices_gate/adjacency_matrix_' + str(k) + '.csv',
            'metricFile0': './Results/AGCRN/'+  str(forecast_len)+ ' Hour Forecast/Metrics/Station_',
            
            'metricFile1': '/split_' + str(k) + '_metrics.txt'

        }
    

    def save_updateMatrix(self, supports, fileDictionary):
        supports_np = supports.detach().cpu().numpy()  # Convert the tensor to a numpy array
        df = pd.DataFrame(supports_np)  # Convert the numpy array to a pandas DataFrame
        current_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(current_dir)
        path = os.path.join(parent_dir, fileDictionary['updateMatrixFile'])
        os.makedirs(os.path.dirname(path), exist_ok=True)

        df.to_csv(fileDictionary['updateMatrixFile'], index=True, header=True) 

    def save_gateMatrix(self, supports, fileDictionary):
        supports_np = supports.detach().cpu().numpy()  # Convert the tensor to a numpy array
        df = pd.DataFrame(supports_np)  # Convert the numpy array to a pandas DataFrame
        current_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(current_dir)
        path = os.path.join(parent_dir, fileDictionary['gateMatrixFile'])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(fileDictionary['gateMatrixFile'], index=True, header=True) 


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.modelConfig
        }
        torch.save(state, self.best_path)
        self.model_logger.info("Saving current best model to " + self.best_path)
        print("Saving current best model to " + self.best_path)


    def test(self, model, modelConfig, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            agcrn = check_point['config']
            model.load_state_dict(state_dict)
            model.to(modelConfig['device']['default'])
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :modelConfig['input_dim']['default']]
                label = target[..., :modelConfig['output_dim']['default']]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        sharedUtil.create_file_if_not_exists(self.fileDictionary["targetFile"])
        sharedUtil.create_file_if_not_exists(self.fileDictionary["predFile"])
        np.save(self.fileDictionary["targetFile"], y_true)
        np.save(self.fileDictionary["predFile"], y_pred)

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
    