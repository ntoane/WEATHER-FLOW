import argparse
import yaml
import HPO.tcnHPO as tcnHPO
import HPO.gwnHPO as gwnHPO
import Train.tcnTrain as tcnTrain
import Train.gwnTrain as gwnTrain
import Evaluation.baselineEval as baselineEval
import Plots.plotter as plotter
# import Visualisations.visualise as visualise

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to YAML config file')
args = parser.parse_args()

# Load the YAML config file which contains all the required settings for platform
with open('configurations/sharedConfig.yaml', 'r') as file:
    sharedConfig = yaml.safe_load(file)

complete = False
def main():
    configOptions = ['train_tcn', 'train_gwn','tune_tcn','tune_gwn','eval_tcn','eval_gwn','vis']
    loop = True
    global complete

    print("............................................................")
    print("Experimental Platform running")

############ Training ###############
    # Train final TCN models using config settings specified
    if sharedConfig['train_tcn']['default']:
        tcnConfig = getSpecificConfig('tcn')
        tcnTrain.train(sharedConfig, tcnConfig)
    
# Train final GWN models using the config settings specified
    if sharedConfig['train_gwn']['default']:
        gwnConfig = getSpecificConfig('gwn')
        gwnTrain.train(sharedConfig, gwnConfig)
  
######### Random Search ##############
    # Random search TCN
    if sharedConfig['tune_tcn']['default']:
        tcnConfig = getSpecificConfig('tcn')
        tcnHPO.hpo(sharedConfig, tcnConfig)

# Random search GWN
    if sharedConfig['tune_gwn']['default']:
        gwnConfig = getSpecificConfig('gwn')
        gwnHPO.hpo(sharedConfig, gwnConfig)

############ Recordings ##############
    # Record metrics for final TCN models
    if sharedConfig['eval_tcn']['default']:
        tcnConfig = getSpecificConfig('tcn')
        baselineEval.TcnEval('TCN', sharedConfig, tcnConfig)
        plotter.create('TCN')

    # Record metrics for final GWN models
    if sharedConfig['eval_gwn']['default']:
        gwnConfig = getSpecificConfig('gwn')
        baselineEval.GwnEval(sharedConfig, gwnConfig)
        plotter.create('GWN')

# ############ Visualisations #############
#     if sharedConfig['vis']['default']:
#         visualise.plot(sharedConfig)
     
############ Else condition #############   
    else :
        if not complete:   
            print("You have not set any of the models in the config.yaml file to True. Please review the config.yaml file again before continuing. :)")
            print("You can make some changes in config.yaml through platform.")
            print("Press a to alter the config file or any other key to exit")
            userInput = input()
            if userInput == 'a':
                print(" ")
                while loop:
                    print('Enter which of the following settings to change or c to continue : train_tcn,train_gwn,tune_tcn,tune_gwn,eval_tcn,eval_gwn,vis')
                    userInput = input()
                    if userInput in configOptions:
                        sharedConfig[userInput]['default'] = not sharedConfig[userInput]['default']
                    elif userInput == 'c':
                        loop = False
                    else:
                        print("Invalid input entered")
                complete = True
                main()
        
def getSpecificConfig(modelName):
    with open('configurations/'+ modelName +'Config.yaml', 'r') as file:
        return yaml.safe_load(file)

if __name__ == '__main__':
    main()

        
    


