from distutils.errors import PreprocessError
from sqlite3 import SQLITE_ALTER_TABLE
import sqlite3
import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import omegaconf
import hydra
import json
from copy import deepcopy

import pandas as pd
from RDkitHelpers import getInchifromSMILES 

def main(args):
    #omegaconf.OmegaConf.set_struct(args, False)

    input = pd.read_csv(args.input)
    if isinstance(args.target,list) or isinstance(args.target,omegaconf.listconfig.ListConfig):
        if len(args.target) == 1:
            targettolookat = args.target[0]
        else:
            raise PreprocessError ('Cannot classify multiple things in one EGAT model.')
    else:
        targettolookat = args.target
    uniqueclasses = input[targettolookat].unique()

    dummy_columns = pd.get_dummies(input[targettolookat], prefix='is', prefix_sep='_', columns=uniqueclasses)
    input = pd.concat([input, dummy_columns], axis=1)

    # Replace NaN values with 0
    input = input.fillna(0)
    input.to_csv(args.input)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--config", type=str, default="config/test.yaml", help="Path to the config file")

    args = parser.parse_args()

    # Load the specified config file
    config = omegaconf.OmegaConf.load(args.config)
    omegaconf.OmegaConf.set_struct(args, False)
    
    # Determine the config_name based on the name of the loaded config file
    file_name = os.path.basename(args.config)
    config_name, _ = os.path.splitext(file_name)

    # Set the config_name for the Hydra function
    hydra.utils.set_config_name(config_name)

    # Run the Hydra function with the merged configuration
    main(config)

    

    

    
