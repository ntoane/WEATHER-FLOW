from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from Execute.clcrnExecute import clcrnExecute
import yaml
import numpy as np
from pathlib import Path
import os
import json
import sys


def main(args):

    with open(args.config_filename, 'r') as f:
        supervisor_config = yaml.safe_load(f)
        path = Path(supervisor_config['log_dir']['default'])/supervisor_config['experiment_name']['default']
        path.mkdir(exist_ok = True, parents = True)
        
        modelName = 'shared'
        with open('configurations/'+ modelName +'Config.yaml', 'r') as file:
            sharedConfig =  yaml.safe_load(file)

        sv_param = os.path.join(path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(supervisor_config, file_obj)
        supervisor = clcrnExecute(sharedConfig,supervisor_config)

        supervisor.execute(supervisor_config)
        supervisor._test_final_n_epoch(1)
        # supervisor._get_time_prediction()
        # supervisor._local_pattern_visual([[1.0, 25.7485, -33.4421]])#, r=0.1, r_resolution=180, phi_resolution=180)


def SetSeed(seed):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch and numpy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2022, type=int, #2021~2025
                        help='Seed for reproducity')
    parser.add_argument('--config_filename', default='./experiments/config_clcrn.yaml', type=str,
                        help='Configuration filename for restoring the model.')
                        
    args = parser.parse_args()
    SetSeed(args.seed)          

    main(args)
