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

    stations = ['ADDO ELEPHANT PARK', 'ALEXANDERBAAI', 'ALIWAL-NORTH PLAATKOP', 'BARKLY-OOS (CAERLEON)',
                'BRANDVLEI', 'CALVINIA WO', 'CAPE TOWN WO', 'DE AAR WO', 'DOHNE - AGR', 'EAST LONDON WO',
                'EXCELSIOR CERES', 'FORT BEAUFORT', 'FRASERBURG', 'GEORGE WITFONTEIN', 'GEORGE WO', 
                'GRAAFF - REINET', 'GRAHAMSTOWN', 'KOINGNAAS', 'LADISMITH', 'LAINGSBURG', 'LANGGEWENS',
                'MALMESBURY', 'MOLTENO RESERVOIR','NOUPOORT','OUDTSHOORN', 'PATENSIE','POFADDER', 
                'PORT ALFRED - AIRPORT','PORT ELIZABETH AWOS', 'PORT ELIZABETH AWS','PORT NOLLOTH','PORTERVILLE', 
                'PRIESKA', 'REDELINGSHUYS-AWS','RIVERSDALE','SOMERSET EAST','SPRINGBOK WO','TWEE RIVIEREN',
                'UITENHAGE','UPINGTON WO', 'VANWYKSVLEI','VIOOLSDRIF - AWS','VREDENDAL','WILLOWMORE','WORCESTER-AWS'
]

    """
    List of points to split data in train, validation, and test sets for walk-forward validation. The first marker, 
    8784 is one year's worth of data, the next step is 3 months of data, and the following step is also 3 months of 
    data, resulting in rolling walk-forward validation where the train size increases each increment, with the 
    validation and test sets each being 3 months' worth of data.
    """

    increment = [100,200,300,8760, 10920, 13106, 15312, 17520, 19704, 21888, 24096, 26304,
                 28464, 30648, 32856, 35064, 37224, 39408, 41616, 43824, 45984,
                 48168, 50376, 52584, 54768, 56952, 59160, 61368, 63528, 65712,
                 67920, 70128, 72288, 74472, 76680, 78888, 81048, 83232, 85440,
                 87648, 89832, 92016, 94224, 96432, 98592, 100776, 102984, 105192,
                 107352, 109536, 111744, 113929]


############ Training ###############
    # Train final TCN models
    if config['train_tcn']['default']:
        tcnTrain.train(stations, increment, config)
    
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
     
############ Else condition #############   
    else :
        print("You have not set any of the models in the config.yaml file to True. Please review the config.yaml file again before continuing. :)")
    
        
    


