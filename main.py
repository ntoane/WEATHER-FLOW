import argparse
import yaml
import HPO.tcnHPO as tcnHPO
import HPO.gwnHPO as gwnHPO
import Train.tcnTrain as tcnTrain
import Train.gwnTrain as gwnTrain
import Evaluation.baselineEval as baselineEval
import plotter
# import Visualisations.visualise as visualise

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to YAML config file')
args = parser.parse_args()

# Load the YAML config file which contains all the required settings for platform
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


complete = False
def main():
    configOptions = ['train_tcn', 'train_gwn','tune_tcn','tune_gwn','eval_tcn','eval_gwn','vis']
    loop = True
    global complete

    print("............................................................")
    print("Experimental Platform running")
############ Training ###############
    # Train final TCN models using config settings specified
    if config['train_tcn']['default']:
        tcnTrain.train(config)
    
# Train final GWN models using the config settings specified
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
        plotter.create('TCN')

    # Record metrics for final GWN models
    if config['eval_gwn']['default']:
        baselineEval.GwnEval(config)
        plotter.create('GWN')

# ############ Visualisations #############
#     if config['vis']['default']:
#         visualise.plot(config)
     
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
                        config[userInput]['default'] = not config[userInput]['default']
                    elif userInput == 'c':
                        loop = False
                    else:
                        print("Invalid input entered")
                complete = True
                main()
        

if __name__ == '__main__':
    main()

        
    


