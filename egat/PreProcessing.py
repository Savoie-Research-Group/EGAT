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
    if args.molecular:
        input['InChIFilter'] = input[args.smiles].apply(getInchifromSMILES)
    else:
        inputsmiles = input[args.smiles].str.split('>>',expand=True)
        inputsmiles.columns = ['Re','Pr']
        inputsmiles['RI'] = inputsmiles.Re.apply(getInchifromSMILES)
        inputsmiles['PI'] = inputsmiles.Pr.apply(getInchifromSMILES)
        input['InChIFilter'] = inputsmiles.RI + '>>' + inputsmiles.PI 
    
    values = input.groupby('InChIFilter').idxmin()
    if isinstance(args.target,list) or isinstance(args.target,omegaconf.listconfig.ListConfig):
        targettolookat = args.target[0]
        bestvalues = values.loc[:,targettolookat]
    else:
        bestvalues = values.loc[:,args.target]
    
    bestvalues = bestvalues.sort_values()
    input = input.loc[bestvalues.tolist()]
    input = input.drop('InChIFilter', axis=1)
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

    

    

    
