import torch.optim as optim
from Models.graphWaveNet import *
import torch as torch

class trainer:
    """
        Initialise GWN Model, place the model on the GPU, intialise optimiser and loss metric for GWN model
    """

    def __init__(self, scaler, supports, aptinit, config):

        self.model = gwnet(config['device']['default'], num_nodes=config['num_nodes']['default'], dropout=config['dropout']['default'], supports=supports,
                           gcn_bool=config['gcn_bool']['default'], addaptadj=config['addaptadj']['default'],
                           aptinit=aptinit, in_dim=config['in_dim']['default'], out_dim=config['seq_length']['default'], residual_channels=config['nhid']['default'],
                           dilation_channels=config['nhid']['default'], skip_channels=config['nhid']['default'] * 8, end_channels=config['nhid']['default'] * 16,
                           layers=config['num_layers']['default'])
        self.model.to(config['device']['default'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate']['default'], weight_decay=config['weight_decay']['default'])
        self.loss = nn.MSELoss(reduction='mean').to(config['device']['default'])
        self.scaler = scaler

    def train(self, trainLoader, config):
        """
        Training logic for the GWN model. Makes predictions on the input supplied, calculates loss and upgrades weights in model.

        Parameters:
            trainLoader - Instance of DataLoader which performs preprocessing operations and an iterator to iterate through the data
            Config - Default configuration settings from config file. 

        Returns:
            train_loss/trainLoader.num_batch - returns the training loss(MSE) across the batches fed into it by the DataLoader
        """
        train_loss = 0
        trainLoader.shuffle()

        for i, (x, y) in enumerate(trainLoader.get_iterator()):
            x = torch.Tensor(x).to(config['device']['default'])
            x = x.transpose(1, 3)
            y = torch.Tensor(y).to(config['device']['default'])
            y = y.transpose(1, 3)
            self.model.train()
            self.optimizer.zero_grad()
            input = nn.functional.pad(x, (1, 0, 0, 0))
            output = self.model(input)
            output = output.transpose(1, 3)
            real = torch.unsqueeze(y[:, 0, :, :], dim=1)
            loss = self.loss(output, real)
            loss.backward()
            self.optimizer.step()
            train_loss += loss

        return (train_loss / trainLoader.num_batch).item()

    def validate(self, validation_loader, config):
        """
        Validation logic for the GWN model. Makes predictions on the input supplied, calculates loss(MSE) without updating weights.

        Parameters:
            trainLoader - Instance of DataLoader which performs preprocessing operations and an iterator to iterate through the data
            Config - Default configuration settings from config file. 

        Returns:
            validation_loss/trainLoader.num_batch - returns the validation loss(MSE) across the batches fed into it by the DataLoader
        """
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            for iter, (x, y) in enumerate(validation_loader.get_iterator()):
                x = torch.Tensor(x).to(config['device']['default'])
                x = x.transpose(1, 3)
                y = torch.Tensor(y).to(config['device']['default'])
                y = y.transpose(1, 3)
                input = nn.functional.pad(x, (1, 0, 0, 0))
                output = self.model(input)
                output = output.transpose(1, 3)
                real = torch.unsqueeze(y[:, 0, :, :], dim=1)
                loss = self.loss(output, real)
                val_loss += loss

            return (val_loss / validation_loader.num_batch).item()

    def test(self, test_loader, config):
        """
        Test logic for the GWN model. Makes predictions on the input supplied, calculates loss(MSE) without updating weights.

        Parameters:
            trainLoader - Instance of DataLoader which performs preprocessing operations and an iterator to iterate through the data
            Config - Default configuration settings from config file. 

        Returns:
            test_loss/trainLoader.num_batch - returns the validation loss(MSE) across the batches fed into it by the DataLoader
            predictions - returns a list of the predictions made by the GWN model on the test set
            targets - returns a list of the test inputs fed into the GWN model
        """
        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            test_loss = 0
            for iter, (x, y) in enumerate(test_loader.get_iterator()):
                x = torch.Tensor(x).to(config['device']['default'])
                x = x.transpose(1, 3)
                y = torch.Tensor(y).to(config['device']['default'])
                y = y.transpose(1, 3)
                input = nn.functional.pad(x, (1, 0, 0, 0))
                output = self.model(input)
                output = output.transpose(1, 3)
                real = torch.unsqueeze(y[:, 0, :, :], dim=1)
                loss = self.loss(output, real)
                test_loss += loss
                predictions.append(output.cpu().detach().numpy())
                targets.append(real.cpu().detach().numpy())
            return (test_loss / test_loader.num_batch).item(), predictions, targets
