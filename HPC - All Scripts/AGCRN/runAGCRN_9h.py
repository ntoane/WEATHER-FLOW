# from Execute.agcrnExecute import train
import yaml
import Logs.Evaluation as Evaluation
from Execute.agcrnExecute import agcrnExecute
from HPO.agcrnHPO import agcrnHPO

###########################################################

with open('configurations/agcrnConfig.yaml', 'r') as file:
    agcrnConfig = yaml.safe_load(file)
with open('configurations/sharedConfig_9h.yaml', 'r') as file:
    sharedConfig = yaml.safe_load(file)




# train(agcrnConfig, agcrnConfig)
agcrn_trainer = agcrnExecute(sharedConfig, agcrnConfig)
agcrn_trainer.execute() 

# agcrn_hpo = agcrnHPO(sharedConfig, agcrnConfig)
# agcrn_hpo.hpo() 

Evaluation.AgcrnEval(agcrnConfig, sharedConfig)
    