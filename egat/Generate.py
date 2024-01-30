from ast import excepthandler
from copy import deepcopy
import sys
import h5py
sys.path.append('utilities')
from taffi_functions import adjmat_to_adjlist,graph_seps,xyz_parse,find_lewis,return_ring_atom
from utility import *
from yarpecule import return_rings
import json
import os,sys,subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import argparse 
import joblib
from joblib import Parallel,delayed
import traceback    
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDistGeom
from rdkit.Chem import inchi
from rxnmapper import RXNMapper
from descriptastorus.descriptors import rdNormalizedDescriptors,rdDescriptors
from pqdm.processes import pqdm
import sqlite3
import multiprocessing
from functools import partial

from graphgenhelperfunctions import *
from openshellRDKit import *
from RDKitAtomMapping import *
from RDKitSMILESGen import *
from RDkitHelpers import *
from LoopFunctionCSV import *


import omegaconf
import hydra


def dump(info,folder):
    table_name = f"{folder.split('/')[-1]}_{info['rxntype']}_{info['Indices']}"
    conn = sqlite3.connect(os.path.join(folder,f"{folder.split('/')[-1]}.db"))
    cursor = conn.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')
    
    # Ensure the table does not exist before creating it
    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
    cursor.execute(f'CREATE TABLE {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT, value TEXT)')

    key_value_pairs = [(key,str(info[key])) for key in list(info.keys())]
    cursor.executemany(f'INSERT INTO {table_name} (key, value) VALUES (?, ?)',
                        key_value_pairs)
    conn.commit()
    conn.close()

def create_tmpargs(info,folder):
    return {'info':info,'folder':folder}


def TurnintoSQL(infolist,folder,num_workers):
    # Connect to SQLite database
    # Base folder path containing subdirectories with JSON files

    tmpargs = Parallel(n_jobs=num_workers)(delayed(create_tmpargs)(info,folder) for info in infolist)
    pqdm(tmpargs,dump,n_jobs=num_workers,argument_type='kwargs')                
    


def Sequential(infolist,folder):
    conn = sqlite3.connect(os.path.join(folder,f"{folder.split('/')[-1]}.db"))
    cursor = conn.cursor()
    for i,info in enumerate(tqdm(infolist, total=len(infolist), smoothing=0.9)):
        try:
            print(info)
            #print(folder.split('/')[-1])
            
            table_name = f"{folder.split('/')[-1]}_{info['rxntype']}_{info['Indices']}"
        except:
            table_name = f"{folder.split('/')[-1]}_{info['Indices']}"
        # Ensure the table does not exist before creating it
        cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
        cursor.execute(f'CREATE TABLE {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT, value TEXT)')

        key_value_pairs = [(key,str(info[key])) for key in list(info.keys())]
        cursor.executemany(f'INSERT INTO {table_name} (key, value) VALUES (?, ?)',
                            key_value_pairs)
    conn.commit()
    conn.close()




def loopfunctionhelper(Rind,args):
    return loopfunctionfromcsv(Rind, args.data_path, args.input, args.target, args.additionals, args.smiles, args.molecular, args.method_mapping, args.folders, args.getradical, args.onlyH, args.getbondrot, args.atom_map, args.removeelementinfo, args.removeneighborcount, args.removeringinfo, args.removereactiveinfo, args.removeformalchargeinfo, args.removearomaticity, args.removechiralinfo, args.removehybridinfo, args.removebondtypeinfo, args.removebondorderinfo, args.removeconjinfo, args.removestereoinfo, args.hasaddons,args.hasnormedaddons,args.datastorage,args.useOld,args.useFullHyb,num_workers=args.num_workers)
            

def create_mapping(Rind,args):
    return [Rind,args]

def main(args):
    #omegaconf.OmegaConf.set_struct(args, False)

    input = pd.read_csv(args.input)
    if not os.path.isdir(args.data_path): os.mkdir(args.data_path)
    
    if args.num_workers == 1:
        pbar = True
        if pbar:
            for Rind in tqdm(input.index.tolist(), total=len(input.index.tolist()), smoothing=0.9):
                loopfunctionfromcsv(Rind,args.data_path,args.input,args.target,args.additionals,args.smiles,args.molecular,args.method_mapping,args.folders,args.getradical,args.onlyH,args.getbondrot,args.atom_map,args.removeelementinfo,args.removeneighborcount,args.removeringinfo,args.removereactiveinfo,args.removeformalchargeinfo,args.removearomaticity,args.removechiralinfo,args.removehybridinfo,args.removebondtypeinfo,args.removebondorderinfo,args.removeconjinfo,args.removestereoinfo,args.hasaddons,args.hasnormedaddons,args.datastorage,args.useOld,args.useFullHyb)
        else:
            for Rind in [input.index.tolist()[0]]:
                loopfunctionfromcsv(Rind,args.data_path,args.input,args.target,args.additionals,args.smiles,args.molecular,args.method_mapping,args.folders,args.getradical,args.onlyH,args.getbondrot,args.atom_map,args.removeelementinfo,args.removeneighborcount,args.removeringinfo,args.removereactiveinfo,args.removeformalchargeinfo,args.removearomaticity,args.removechiralinfo,args.removehybridinfo,args.removebondtypeinfo,args.removebondorderinfo,args.removeconjinfo,args.removestereoinfo,args.hasaddons,args.hasnormedaddons,args.datastorage,args.useOld,args.useFullHyb)
    else:
        try:
            print(args.data_path)
            print(type(args.data_path))
            print(args.data_path.split('/'))
            print('OBTAINING LIST OF DICTIONARIES')
            print()
            
            #mapping = [[Rind,args] for Rind in input.index.tolist()]
            mapping = Parallel(n_jobs=args.num_workers)(delayed(create_mapping)(Rind,args) for Rind in input.index.tolist())
            infolist = pqdm(mapping,loopfunctionhelper,n_jobs=args.num_workers,argument_type='args')
            print()
            print('DUMPING INTO SQL')
            print()
            #TurnintoSQL(infolist,args.data_path,args.num_workers)
            Sequential(infolist,args.data_path)
            

        except Exception as e:
            import traceback
            traceback.print_exc()
            print('Parallelization Fails')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--config", type=str, default="config/test.yaml", help="Path to the config file")

    args = parser.parse_args()

    # Load the specified config file
    config = omegaconf.OmegaConf.load(args.config)
    omegaconf.OmegaConf.set_struct(config, False)
    
    # Determine the config_name based on the name of the loaded config file
    file_name = os.path.basename(args.config)
    config_name, _ = os.path.splitext(file_name)

    # Set the config_name for the Hydra function
    #hydra.utils.set_config_name(config_name)

    # Run the Hydra function with the merged configuration
    main(config)