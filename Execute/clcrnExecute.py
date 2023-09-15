import os
import time
import numpy as np
import torch
from Utils.CLCRN_Utils import dataloader ,createData, utils
import yaml
from Models.CLCRN.clcnn import CLCRNModel
from Utils.CLCRN_Utils.loss import masked_mae_loss, masked_mse_loss, masked_smape_loss
from tqdm import tqdm
from pathlib import Path
from Execute.modelExecute import modelExecute
import pickle
import warnings
torch.set_num_threads(4)

def exists(val):
    return val is not None

class clcrnExecute(modelExecute):
    
    def __init__(self,sharedConfig,clcrnConfig):
        super().__init__('clcrn', sharedConfig, clcrnConfig)
        # self._device = torch.device("cuda:{}".format(self.modelConfig['gpu']['default']) if torch.cuda.is_available() else "cpu") 
        # self.modelConfig['device']['default']) if torch.cuda.is_available() else "cpu") 
        self._device = torch.device("cpu")
        self.horizon = self.sharedConfig['horizons']['default']
        self.max_grad_norm = self.modelConfig['max_grad_norm']['default']

        # logging.
        self._experiment_name = self.modelConfig['experiment_name']['default']
        self._log_dir = self._get_log_dir(self)

        self._model_name = self.modelConfig['model_name']['default']

        log_level = 'INFO'
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self.increments = self.sharedConfig['increment']['default']

        self.num_nodes = int(self.modelConfig['num_nodes']['default'])
        self.input_dim = int(self.modelConfig['input_dim']['default'])
        self.seq_len = int(self.modelConfig['seq_len']['default'])  # for the encoder
        self.output_dim = int(self.modelConfig['output_dim']['default'])
        self.use_curriculum_learning = bool(self.modelConfig['use_curriculum_learning']['default'])
        
        self.save_dir = self.modelConfig['save_dir']['default']
        
        self._epoch_num = self.modelConfig['epoch']['default']
        
    @staticmethod
    def _get_log_dir(self):
        log_dir = Path(self.modelConfig['log_dir']['default'])/'{} Hour Forecast'.format(self.horizon[0])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):

        model_path = Path(self.save_dir)/'{} Hour Models'.format(self.horizon[0])
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        defaults = {}
        for key, value in self.modelConfig.items():
            defaults[key] = value['default']

        modelDict = dict(defaults)
        modelDict['model_state_dict']= self.model.state_dict()
        modelDict['epoch'] = epoch
        torch.save(modelDict, model_path/('{}_epo{}.tar'.format(self.horizon[0],epoch)))
        self._logger.info("Saved model at {}".format(epoch))
        return '{}_epo{}.tar'.format(self.horizon[0],epoch)

    def load_model(self, epoch_num):

        self._setup_graph()
        model_path = Path(self.save_dir)/'{} Hour Models'.format(self.horizon[0])
        assert os.path.exists(model_path/('{}_epo{}.tar'.format(self.horizon[0],epoch_num))), 'Weights at epoch %d not found' % epoch_num
        checkpoint = torch.load(model_path/('{}_epo{}.tar'.format(self.horizon[0],epoch_num)), map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.model = self.model.eval()

            val_iterator = self._data['val_loader']

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.model(x)
                break

    def execute(self):
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
                self.getMatrix('s')

                self._logger.info("Model created for horizon : " + str(h) + " split " + str(self.increments[s]))
                self._train(h,self.modelConfig['base_lr']['default'], self.modelConfig['steps']['default'])
            

    def evaluate(self, dataset, batches_seen, epoch_num, load_model=False, steps=None):

        if load_model == True:
            self.load_model(epoch_num)

        with torch.no_grad():
            self.model = self.model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)]
            losses = []
            y_truths = []
            y_preds = []

            MAE_metric = masked_mae_loss
            MSE_metric = masked_mse_loss
            SMAPE_metric = masked_smape_loss
            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.model(x)
                loss, y_true, y_pred = self._compute_loss(y, output)
                
                losses.append(loss.item())
                y_truths.append(y_true.cpu())
                y_preds.append(y_pred.cpu())

            mean_loss = np.mean(losses)
            y_preds = torch.cat(y_preds, dim=1)
            y_truths = torch.cat(y_truths, dim=1)

            loss_mae = MAE_metric(y_preds, y_truths).item()
            loss_mse = MSE_metric(y_preds, y_truths).item()
            loss_mape = SMAPE_metric(y_preds, y_truths).item()
            dict_out = {'prediction': y_preds, 'truth': y_truths}
            dict_metrics = {}
            if exists(steps):
                for step in steps:
                    assert(step <= y_preds.shape[0]), ('the largest step is should smaller than prediction horizon!!!')
                    y_p = y_preds[:step, ...]
                    y_t = y_truths[:step, ...]
                    dict_metrics['mae_{}'.format(step)] = MAE_metric(y_p, y_t).item()
                    dict_metrics['mse_{}'.format(step)] = MSE_metric(y_p, y_t).sqrt().item()
                    dict_metrics['smape_{}'.format(step)] = SMAPE_metric(y_p, y_t).item()

            return loss_mae, loss_mse, loss_mape, dict_out, dict_metrics

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

                x, y = self._prepare_data(x, y)
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr_scheduler.step()
            # lr_scheduler.step()
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

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        return x.to(self._device), y.to(self._device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3).float()
        y = y.permute(1, 0, 2, 3).float()
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        for out_dim in range(self.output_dim):
            y_true[...,out_dim] = self.standard_scaler.inverse_transform(y_true[...,out_dim])
            y_predicted[...,out_dim] = self.standard_scaler.inverse_transform(y_predicted[...,out_dim])

        return masked_mae_loss(y_predicted, y_true), y_true, y_predicted

    def _convert_scale(self, y_true, y_predicted):
        for out_dim in range(self.output_dim):
            y_true[...,out_dim] = self.standard_scaler.inverse_transform(y_true[...,out_dim])
            y_predicted[...,out_dim] = self.standard_scaler.inverse_transform(y_predicted[...,out_dim])

        return y_true, y_predicted
        
    def _prepare_x(self, x):
        x = x.permute(1, 0, 2, 3).float()
        return x.to(self._device)
    
    def _test_final_n_epoch(self, n=1, steps=[]):
        model_path = Path(self.save_dir)/'{} Hour Models'.format(self.horizon[0])
        model_list = os.listdir(model_path)
        import re

        epoch_list = []
        for filename in model_list:
            epoch_list.append(int(re.search(r'\d+', filename[4:]).group()))

        epoch_list = np.sort(epoch_list)[-n:]
        for i in range(n):
            epoch_num = epoch_list[i]
            mean_score, mean_loss_mse, mean_loss_mape, _, dict_metrics = self.evaluate('test', 0, epoch_num, load_model=True, steps=steps)
            message = "Loaded the {}-th epoch.".format(epoch_num) + \
                " MAE : {}".format(mean_score), "MSE : {}".format(np.sqrt(mean_loss_mse)), "SMAPE : {}".format(mean_loss_mape)
            self._logger.info(message)
            message = "Metrics in different steps: {}".format(dict_metrics)
            self._logger.info(message)
    
    def _get_time_prediction(self):
        import copy
        
        h =  self.sharedConfig['horizons']['default']
        h = h[0]
        self._logger.info("Storing actuals vs predictions")
        path =  self.modelConfig['dataset_dir']['default'] + '/horizon_{}'.format(h)
        position_path = self.modelConfig['dataset_dir']['default'] + '/horizon_{}/position_info.pkl'.format(h)
        # position_path = '/data/test/position_info.pkl'
  
        _data = dataloader.load_dataset(path, self.increments[0] ,self.modelConfig['batch_size']['default'],position_path)
        test_loader = _data['test_loader']
        y_preds = []
        y_trues = []
        
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                
                x, y = self._prepare_data(x, y)
               
                output = self.model(x)
                
                y = y.permute(1, 0, 2, 3)
                output = output.permute(1, 0, 2, 3)
                y_preds.append(output)
                
                y_trues.append(y)

            y_preds = torch.cat(y_preds, 0).squeeze(dim=1).cpu().numpy()
            y_trues = torch.cat(y_trues, 0).squeeze(dim=1).cpu().numpy()

            import pickle
            with open('{}/actuals.pkl'.format(self._log_dir), "wb") as f:
                smapeData = {'y_trues': y_trues}
                pickle.dump(smapeData, f, protocol = 4)

            with open('{}/predictions.pkl'.format(self._log_dir), "wb") as f:
                smapeData = {'y_preds': y_preds}
                pickle.dump(smapeData, f, protocol = 4)
        print("Storing actuals vs predicted complete")

    def getLogger(self):
        return self._logger

      
    def getMatrix(self,mode):
       
        kernel = self.model.get_kernel()
        coor = kernel.getCoor()
        matrix = kernel.conv_kernel(coor,True)
        
        matrix = matrix.tolist()

        with open('{}/adjMatrix_{}.pkl'.format(self._log_dir,mode), "wb") as f:
            smapeData = {'adj_matrix': matrix}
            pickle.dump(smapeData, f, protocol = 4)

        self._logger.info("Matrix saved in Logs")


        
