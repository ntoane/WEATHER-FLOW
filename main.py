import argparse
import yaml
import HPO.tcnHPO as tcnHPO
import HPO.gwnHPO as gwnHPO
import Train.tcnTrain as tcnTrain
import Train.gwnTrain as gwnTrain
import Evaluation.baselineEval as baselineEval
import Visualisations.visualise as visualise

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to YAML config file')
args = parser.parse_args()


# Load the YAML config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if __name__ == '__main__':
    # Access the configuration default values from config file
    geo_vis = config['geoVis']['default']
    model_vis = config['modelVis']['default']
    horizon_vis = config['horizonVis']['default']
    split_vis = config['splitVis']['default']
    num_configs = config['num_configs']['default']
    tune_tcn = config['tune_tcn']['default']
    tune_gwn = config['tune_gwn']['default']
    train_tcn = config['train_tcn']['default']
    train_gwn = config['train_gwn']['default']
    eval_tcn = config['eval_tcn']['default']
    eval_gwn = config['eval_gwn']['default']
    n_stations = config['n_stations']['default']
    n_split = config['n_split']['default']
    device = config['device']['default']
    data = config['data']['default']
    adjdata = config['adjdata']['default']
    adjtype = config['adjtype']['default']
    gcn_bool = config['gcn_bool']['default']
    aptonly = config['aptonly']['default']
    addaptadj = config['addaptadj']['default']
    randomadj = config['randomadj']['default']
    seq_length = config['seq_length']['default']
    lag_length = config['lag_length']['default']
    nhid = config['nhid']['default']
    in_dim = config['in_dim']['default']
    num_nodes = config['num_nodes']['default']
    num_layers = config['num_layers']['default']
    batch_size = config['batch_size']['default']
    patience = config['patience']['default']
    learning_rate = config['learning_rate']['default']
    dropout = config['dropout']['default']
    weight_decay = config['weight_decay']['default']
    epochs = config['epochs']['default']
    save = config['save']['default']
    adjTrainable = config['adjTrainable']['default']
    use_sparse = config['use_sparse']['default']
    nlayers = config['nlayers']['default']
    features = config['features']['default']
    nbatch_size = config['nbatch_size']['default']
    nbatches = config['nbatches']['default']
    nhidden = config['nhidden']['default']
    nsteps = config['nsteps']['default']
    mask_len = config['mask_len']['default']
    forecast = config['forecast']['default']
    rank = config['rank']['default']
    display_step = config['display_step']['default']
    between_lr_updates = config['between_lr_updates']['default']
    learningRate = config['learningRate']['default']
    lr_factor = config['lr_factor']['default']

    # list of all weather stations
    stations = ['Atlantis', 'Calvinia WO', 'Cape Columbine', 'Cape Point',
                'Cape Town - Royal Yacht Club', 'Cape Town Slangkop', 'Excelsior Ceres', 'Hermanus',
                'Jonkershoek', 'Kirstenbosch', 'Ladismith', 'Molteno Resevoir', 'Paarl',
                'Porterville', 'Robben Island', 'Robertson', 'SA Astronomical Observatory',
                'Struisbaai', 'Tygerhoek', 'Wellington', 'Worcester AWS']

    """
    List of points to split data in train, validation, and test sets for walk-forward validation. The first marker, 
    8784 is one year's worth of data, the next step is 3 months of data, and the following step is also 3 months of 
    data, resulting in rolling walk-forward validation where the train size increases each increment, with the 
    validation and test sets each being 3 months' worth of data.
    """
    increment = [8784, 10944, 13128, 15336, 17544, 19704, 21888,
                 24096, 26304, 28464, 30648, 32856, 35064, 37248,
                 39432, 41640, 43848, 46008, 48192, 50400, 52608,
                 54768, 56952, 59160, 61368, 63528, 65712, 67920, 70128]

    """
    Adjusted increment list seen above for WGN model. A number of steps are removed when shifting the time-series up
    to process the data into input-output pairs. args.forecast = forecast length(24 hours).
    """
    wgn_increment = [8784, 10944, 13128, 15336, 17544, 19704, 21888,
                     24096, 26304, 28464, 30648, 32856, 35064, 37248,
                     39432, 41640, 43848, 46008, 48192, 50400, 52608,
                     54768, 56952, 59160, 61368, 63528, 65712, 67920, 70128 -  config['forecast']['default'] ] 

############ Training ###############
    # Train final TCN models
    if config['train_tcn']['default']:
        tcnTrain.train(stations, increment)
    
    # Train final GWN models
    if config['train_gwn']['default']:
        gwnTrain.train(increment, config)
  
######### Random Search ##############
    # Random search TCN
    if config['tune_tcn']['default']:
        tcnHPO.hpo(stations, increment, config)

    # Random search GWN
    if config['tune_gwn']['default']:
        gwnHPO.hpo(increment, config)

############ Recordings ##############
    # Record metrics for final TCN models
    if config['eval_tcn']['default']:
        baselineEval.TcnEval(stations, 'TCN')

    # Record metrics for final GWN models
    if config['eval_gwn']['default']:
        baselineEval.GwnEval(stations, config)

############ Visualisations #############
    if config['vis']['default']:
        visualise.plot(config)


