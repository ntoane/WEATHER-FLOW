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
from Execute.modelExecute import modelExecute
from Logs.modelLogger import modelLogger 
from HPO.modelHPO import modelHPO

class agcrnHPO(modelHPO):
    
    def __init__(self, sharedConfig, modelConfig):
        super().__init__('agcrn', sharedConfig, modelConfig)

    def hpo(self):
        #setup file to save parameters to
        param_path = 'HPO/Best Parameters/AGCRN/'
        if not os.path.exists(param_path):
                        os.makedirs(param_path)
        f = open(param_path + "configurations.txt", 'w')

        #setup HPO logs
        log_path = 'Logs/AGCRN/HPO/'
        os.makedirs(log_path, exist_ok=True)
        log_file = log_path + 'agcrn_all_stations.txt'
        self.model_logger = modelLogger('agcrn', 'all_stations', log_file, log_enabled=True)
        
        #run HPO for 24 hour horizon only
        forecast_len = 24
        for config in range(2):
            #generate and print random parameters
            self.model_logger.info("Generating random parameters for AGCRN")
            print("Generating random parameters for AGCRN")
            self.modelConfig = agcrnUtil.generateRandomParameters(self.modelConfig)
            print('This is the HPO configuration: \n',
                          'Lag_length - ', self.modelConfig['lag']['default'], '\n',
                          'Rnn Units - ', self.modelConfig['rnn_units']['default'], '\n',
                          'Layers - ', self.modelConfig['num_layers']['default'], '\n',
                          'Batch Size - ', self.modelConfig['batch_size']['default'], '\n',
                          'Epochs - ', self.modelConfig['epochs']['default'])
            
            #perform model and data prep and train model with configurations
            split=0
            print("Training AGCRN for horizons:"+ str(forecast_len) + "   split:"+str(split))
            self.prep_split_data(forecast_len, split)
            self.initialise_model()
            totLoss = self.execute_split() 

            #save configurations and loss to file
            f.write(    'Lag - ' + str(self.modelConfig['lag']['default']) + '\n' +
                        'Rnn Units - ' + str(self.modelConfig['rnn_units']['default']) + '\n' +
                        'Layers - '+ str(self.modelConfig['num_layers']['default'])+ '\n'+
                        'Batch Size - '+ str(self.modelConfig['batch_size']['default']) + '\n'+
                        'Epochs - '+ str(self.modelConfig['epochs']['default']) + '\n' +
                        'Total Loss: ' + str(totLoss) + '\n\n')
        f.close()


    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):

                data = data[..., :self.modelConfig['input_dim']['default']]  #need to make 6 (inputs)
                label = target[..., :self.modelConfig['output_dim']['default']]   #need to make 1 (output supposed to predict)
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
            if self.modelConfig['teacher_forcing']['default']:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.modelConfig['tf_decay_steps']['default'])
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F

            output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)

            loss = self.loss(output.cpu(), label)
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


    def execute_split(self):
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

            # save the best state
            if best_state == True:
                self.model_logger.info('*********************************Current best model saved!')
                print('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.model_logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        return best_loss


    def initialise_model(self):
        DEVICE = 'cuda:0'
        agcrnUtil.init_seed(self.modelConfig['seed']['default'])

        if torch.cuda.is_available():
            torch.cuda.set_device(int(DEVICE))
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

        

    def prep_split_data(self, horizon, k):
        increments = self.sharedConfig['increment']['default']
        n_stations = self.sharedConfig['n_stations']['default']
        self.train_loader, self.val_loader, self.test_loader, self.scaler = agcrnUtil.get_dataloader(horizon, k, increments, n_stations, self.modelConfig,
                                                                    normalizer=self.modelConfig['normalizer']['default'],
                                                                    tod=self.modelConfig['tod']['default'], dow=False,
                                                                    weather=False, single=False)

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)


    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))

    def train_model(self, X_train, Y_train, X_val, Y_val, cfg, saveFile, n_ft):
        print("reachL")
    