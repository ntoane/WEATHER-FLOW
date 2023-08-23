import argparse
import yaml
from HPO.tcnHPO import TCNHPO
from HPO.gwnHPO import GWNHPO
from Execute.agcrnExecute import agcrnExecute
from Execute.tcnExecute import tcnExecute
from Execute.gwnExecute import gwnExecute
import Plots.plotter as plotter
# import Visualisations.visualise as visualise
# from Logs.Evaluation import Evaluation
import Logs.Evaluation as Evaluation

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='path to YAML config file')
args = parser.parse_args()

# Load the YAML config file which contains all the required settings for platform
with open('configurations/sharedConfig3.yaml', 'r') as file:
    sharedConfig3 = yaml.safe_load(file)

complete = False
def main():
    configOptions = ['train_tcn', 'train_gwn','tune_tcn','tune_gwn','eval_tcn','eval_gwn','train_agcrn', 'eval_agcrn','vis']
    loop = True
    global complete

    print("............................................................")
    print("Experimental Platform running") 

############ Training ###############
   
    
    # Train final GWN models using the config settings specified
    if sharedConfig3['train_gwn']['default'] or args.mode == configOptions[1]:
        gwnConfig = getSpecificConfig('gwn')
        gwn_trainer = gwnExecute(sharedConfig3, gwnConfig)
        gwn_trainer.execute()
        complete = True


        
def getSpecificConfig(modelName):
    with open('configurations/'+ modelName +'Config.yaml', 'r') as file:
        return yaml.safe_load(file)

if __name__ == '__main__':
    main()