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
with open('configurations/sharedConfig.yaml', 'r') as file:
    sharedConfig = yaml.safe_load(file)

complete = False
def main():
    configOptions = ['train_tcn', 'train_gwn','tune_tcn','tune_gwn','eval_tcn','eval_gwn','train_agcrn', 'eval_agcrn','vis']
    loop = True
    global complete

    print("............................................................")
    print("Experimental Platform running") 

############ Training ###############
    # Train final AGCRN models using config settings specified
    if sharedConfig['train_agcrn']['default'] or args.mode == configOptions[6]:
        agcrnConfig = getSpecificConfig('agcrn')
        agcrn_trainer = agcrnExecute(sharedConfig, agcrnConfig)
        agcrn_trainer.execute()
        complete = True


    # Train final TCN models using config settings specified
    if sharedConfig['train_tcn']['default'] or args.mode == configOptions[0]:
        tcnConfig = getSpecificConfig('tcn')
        tcn_trainer = tcnExecute(sharedConfig, tcnConfig)
        tcn_trainer.execute()
        complete = True
    
    # Train final GWN models using the config settings specified
    if sharedConfig['train_gwn']['default'] or args.mode == configOptions[1]:
        gwnConfig = getSpecificConfig('gwn')
        gwn_trainer = gwnExecute(sharedConfig, gwnConfig)
        gwn_trainer.execute()
        complete = True

######### Random Search ##############
    # Random search TCN
    if sharedConfig['tune_tcn']['default'] or args.mode == configOptions[2]:
        tcnConfig = getSpecificConfig('tcn')
        tcnHPO = TCNHPO(sharedConfig, tcnConfig)
        tcnHPO.hpo()
        complete = True


# Random search GWN
    if sharedConfig['tune_gwn']['default'] or args.mode == configOptions[3]:
        gwnConfig = getSpecificConfig('gwn')
        gwnHPO = GWNHPO(sharedConfig, gwnConfig)
        gwnHPO.hpo()
        complete = True

############ Recordings ##############
    # Record metrics for final TCN models
    if sharedConfig['eval_tcn']['default'] or args.mode == configOptions[4]:
        tcnConfig = getSpecificConfig('tcn')
        Evaluation.TcnEval(tcnConfig, sharedConfig)
        complete = True
        # plotter.create('TCN',sharedConfig)

    # Record metrics for final GWN models
    if sharedConfig['eval_gwn']['default'] or args.mode == configOptions[5]:
        gwnConfig = getSpecificConfig('gwn')
        Evaluation.GwnEval(gwnConfig, sharedConfig)
        complete = True
        # plotter.create('GWN', sharedConfig)

    # Record metrics for final AGCRN models
    if sharedConfig['eval_agcrn']['default'] or args.mode == configOptions[7]:
        agcrnConfig = getSpecificConfig('agcrn')
        Evaluation.AgcrnEval(agcrnConfig, sharedConfig)
        complete = True
        # plotter.create('AGCRN', sharedConfig)





# ############ Visualisations #############
    # if sharedConfig['vis']['default'] or args.mode == configOptions[8]:
        # visualise.plot(sharedConfig)

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
                    print('Enter which of the following settings to change or c to continue : train_tcn,train_gwn,tune_tcn,tune_gwn,eval_tcn,eval_gwn,train_agcrn,eval_agcrn,vis')
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