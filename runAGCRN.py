import yaml
import Logs.Evaluation as Evaluation
from Execute.agcrnExecute import agcrnExecute
from HPO.agcrnHPO import agcrnHPO

###########################################################

with open('configurations/agcrnConfig.yaml', 'r') as file:
    agcrnConfig = yaml.safe_load(file)
with open('configurations/sharedConfig.yaml', 'r') as file:
    sharedConfig = yaml.safe_load(file)


agcrn_hpo = agcrnHPO(sharedConfig, agcrnConfig)
agcrn_hpo.hpo() 

agcrn_trainer = agcrnExecute(sharedConfig, agcrnConfig)
agcrn_trainer.execute() 

Evaluation.AgcrnEval(agcrnConfig, sharedConfig)
    