import os
import pdb

from easydict import EasyDict as edict
import yaml
import argparse
import numpy as np
import random
import torch

from optimize_final import CASA_Trainer
from clip_retrieve import retrieve_single

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config_file', type=str, default='config/synthetic.yaml', help='path to config file')

    args = parser.parse_args()

    config_file = args.config_file
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
    set_seed(config.seed)

    ### get retrieve animal
    retrieve_single(config,config_file)

    ### optimize
    trainer = CASA_Trainer(config)
    trainer.optimize()

if __name__ =='__main__':
    main()