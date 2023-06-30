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

############ Training ###############
    # Train final TCN models
    if config['train_tcn']['default']:
        tcnTrain.train(config)
    
    # Train final GWN models
    if config['train_gwn']['default']:
        gwnTrain.train(config)
  
######### Random Search ##############
    # Random search TCN
    if config['tune_tcn']['default']:
        tcnHPO.hpo(config)

    # Random search GWN
    if config['tune_gwn']['default']:
        gwnHPO.hpo(config)

############ Recordings ##############
    # Record metrics for final TCN models
    if config['eval_tcn']['default']:
        baselineEval.TcnEval('TCN', config)

    # Record metrics for final GWN models
    if config['eval_gwn']['default']:
        baselineEval.GwnEval(config)

############ Visualisations #############
    if config['vis']['default']:
        visualise.plot(config)
     
############ Else condition #############   
    else :
        print("You have not set any of the models in the config.yaml file to True. Please review the config.yaml file again before continuing. :)")
    
        
    


