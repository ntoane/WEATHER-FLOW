import os
import time
import numpy as np
import torch
from Utils.CLCRN_Utils import dataloader ,createData, utils
import yaml
from Models.CLCRN.clcnn import CLCRNModel
from Utils.CLCRN_Utils.loss import masked_mae_loss, masked_mse_loss, masked_mape_loss, masked_smape_loss
from tqdm import tqdm
from pathlib import Path
# from Execute.modelExecute import modelExecute
from Logs import Evaluation
import pickle
# from Logs.modelLogger import modelLogger
from HPO.modelHPO import modelHPO
import pandas as pd
import Utils.gwnUtils as util
import Utils.sharedUtils as sharedUtil

class CLCRNHPO(modelHPO):
    def __init__(self, sharedConfig, clcrnConfig):
        super().__init__('clcrn', sharedConfig, clcrnConfig)

    def hpo(self):
        increment = self.sharedConfig['increment']['default']
        # data = self.prepare_data() 
        h = 3
        path =  self.modelConfig['dataset_dir']['default'] + '/horizon_{}'.format(h)
        position_path = self.modelConfig['dataset_dir']['default'] + '/horizon_{}/position_info.pkl'.format(h)
        data = dataloader.load_dataset(path, increment[0] ,self.modelConfig['batch_size']['default'],position_path)
        textFile = 'HPO/Best Parameters/CLCRN/configurations.txt'
        f = open(textFile, 'w')

        self._model_name = self.modelConfig['model_name']['default']

        # best_mse = np.inf

        num_splits = 2
        for i in range(self.sharedConfig['num_configs']['default']):
            config = util.generateRandomParameters(self.modelConfig)
            valid_config = True
            targets = []
            preds = []

            for k in range(num_splits):
                modelFile = 'Garage/HPO Models/CLCRN/{} Hour Models/{}_epo1.tar'
                n_stations= int(self.sharedConfig['n_stations']['default'])
                # data_sets, split = split_data(data, increment, k, n_stations)
                # split = [increment[k] * n_stations, increment[k + 1] * n_stations, increment[k + 2] * n_stations]
                # data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]

                try:
                    print('This is the HPO configuration: \n',
                          'Dropout - ', self.modelConfig['dropout']['default'], '\n',
                          'Lag_length - ', self.modelConfig['lag_length']['default'], '\n',
                          'Hidden Units - ', self.modelConfig['hidden_units']['default'], '\n',
                          'Layers - ', self.modelConfig['layer_num']['default'], '\n',
                          'Batch Size - ', self.modelConfig['batch_size']['default'], '\n',
                          'Epochs - ', self.modelConfig['epochs']['default'])

                    self._train(h,self.modelConfig['base_lr']['default'], self.modelConfig['steps']['default'])
            
                except Warning:
                    valid_config = False
                    break
                
                # clcrn_trainer = clcrnExecute(sharedConfig, clcrnConfig)
                self.train_model()
                self.get_time_prediction()
                

                log_dir = Evaluation.get_log_dir(self.sharedConfig,self.modelConfig)
                
                true_file = '{}/actuals.pkl'.format(log_dir,h)
                predict_file = '{}/predictions.pkl'.format(log_dir,h)

                with open(true_file, 'rb') as f:
                # Load the content of the pickle file
                    trueVals = pickle.load(f)
                with open(predict_file, 'rb') as f:
                # Load the content of the pickle file
                    predVals = pickle.load(f)

            if valid_config:
                mse = masked_mse_loss(np.concatenate(np.array(targets, dtype=object)),
                                  np.concatenate(np.array(preds, dtype=object)))
                if mse < best_mse:
                    best_cfg = config
                    best_mse = mse

        f.write('This is the best configuration ' + str(best_cfg) + ' with an MSE of ' + str(best_mse))
        f.close()
        self.model_logger.info('clcrnHPO : clcrn best configuration found = ' +str(best_cfg) + ' with an MSE of ' + str(best_mse))
        self.model_logger.info('clcrnHPO : clcrn HPO finished at all stations :)')

    def train_model(self):
        splitsLen = len(self.increments)
        horizon = self.horizon  # for the decoder
        for h in horizon :
            for s in range(0,splitsLen-2):
                fPath = self.modelConfig['data']['default']
                outputDir = self.modelConfig['dataset_dir']['default']
                self.standard_scaler = createData.main(h,self.modelConfig,fPath,outputDir,self.increments[s:s+3])

                path =  self.modelConfig['dataset_dir']['default'] + '/horizon_{}'.format(h)
                position_path = self.modelConfig['dataset_dir']['default'] + '/horizon_{}/position_info.pkl'.format(h)
                self._data = dataloader.load_dataset(path, self.increments[s] ,self.modelConfig['batch_size']['default'],position_path)
                self.sparse_idx = torch.from_numpy(self._data['kernel_info']['sparse_idx']).long().to(self._device)
                self.location_info = torch.from_numpy(self._data['kernel_info']['MLP_inputs']).float().to(self._device)
                self.geodesic = torch.from_numpy(self._data['kernel_info']['geodesic']).float().to(self._device)
                self.angle_ratio = torch.from_numpy(self._data['kernel_info']['angle_ratio']).float().to(self._device)
                print('==========================================')
                print(self.angle_ratio)

                if self._model_name == 'CLCRN':
                    model = CLCRNModel(
                        self.location_info, 
                        self.sparse_idx, 
                        self.geodesic, 
                        self.modelConfig,
                        h,
                        self.angle_ratio, 
                        logger=self._logger, 
                        )
                else:
                    print('The method is not provided.')
                    exit()
                self.model = model.to(self._device)

                self._logger.info("Model created for horizon : " + str(h) + " split " + str(self.increments[s]))
                self._train(h,self.modelConfig['base_lr']['default'], self.modelConfig['steps']['default'])
    
    def _train(self, horizon,  base_lr,
               steps, patience=25, epochs=1, lr_decay_ratio=0.1, log_every=1, save_model=True,
               test_every_n_epochs=1, epsilon=1e-8):
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)
        epochs = self.modelConfig['epochs']['default']
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...')
        num_batches = len(self._data['train_loader'])
        self._logger.info("num_batches:{}".format(num_batches))

        best_epoch=0
        batches_seen = num_batches * self._epoch_num
        for epo in range(self._epoch_num, epochs):
            
            epoch_num = epo + 1
            self.model = self.model.train()

            train_iterator = self._data['train_loader']
            losses = []

            start_time = time.time()
            progress_bar =  tqdm(train_iterator,unit="batch")

            for _, (x, y) in enumerate(progress_bar): 
                
                optimizer.zero_grad()

                x, y = self.prepare_data(x, y)
                output = self.model(x, y, batches_seen = batches_seen)
                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

                loss, y_true, y_pred = self._compute_loss(y, output)
                
                progress_bar.set_postfix(training_loss=loss.item())
                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
            
    
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

    
            val_loss, val_loss_mse, val_loss_mape, _, __ = self.evaluate(dataset='val', batches_seen=batches_seen, epoch_num=epoch_num)

            end_time = time.time()

            if (epoch_num % log_every) == 0:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_smape: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, val_loss_mape, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == 0:
                test_loss, test_loss_mse, test_loss_smape, _, __ = self.evaluate(dataset='test', batches_seen=batches_seen, epoch_num=epoch_num)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)
                message = 'Epoch [{}/{}] ({}) test_mse: {:.4f}, test_smape: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                            test_loss_mse, test_loss_smape, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    best_epoch=epoch_num
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break        

    
def prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        return x.to(self._device), y.to(self._device)

def split_data(self, data, increment, k, n_stations):
        split = [increment[k] * n_stations, increment[k + 1] * n_stations, increment[k + 2] * n_stations]
        data_sets = [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]
        return data_sets, split

