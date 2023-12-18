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

from graphgenhelperfunctions import *
from openshellRDKit import *
from RDKitAtomMapping import *
from RDKitSMILESGen import *
from RDkitHelpers import *
from LoopFunctionCSV import *


import omegaconf
import hydra

def main(args):
    #omegaconf.OmegaConf.set_struct(args, False)

    input = pd.read_csv(args.input)
    if not os.path.isdir(args.data_path): os.mkdir(args.data_path)
    if args.num_workers == 1:
        for Rind in tqdm(input.index.tolist(), total=len(input.index.tolist()), smoothing=0.9):
            loopfunctionfromcsv(Rind,args.data_path,args.input,args.target,args.additionals,args.smiles,args.molecular,args.method_mapping,args.folders,args.getradical,args.onlyH,args.getbondrot,args.atom_map,args.removeelementinfo,args.removeneighborcount,args.removeringinfo,args.removereactiveinfo,args.removeformalchargeinfo,args.removearomaticity,args.removechiralinfo,args.removehybridinfo,args.removebondtypeinfo,args.removebondorderinfo,args.removeconjinfo,args.removestereoinfo,args.hasaddons,args.hasnormedaddons)
    else:
        try:
            with tqdm(total=len(input.index.tolist()), desc="Processing", unit="Rind") as pbar:
                Parallel(n_jobs=args.num_workers)(delayed(loopfunctionfromcsv)(Rind, args.data_path, args.input, args.target, args.additionals, args.smiles, args.molecular, args.method_mapping, args.folders, args.getradical, args.onlyH, args.getbondrot, args.atom_map, args.removeelementinfo, args.removeneighborcount, args.removeringinfo, args.removereactiveinfo, args.removeformalchargeinfo, args.removearomaticity, args.removechiralinfo, args.removehybridinfo, args.removebondtypeinfo, args.removebondorderinfo, args.removeconjinfo, args.removestereoinfo, args.hasaddons,args.hasnormedaddons) for Rind in input.index.tolist())
                pbar.update(1)
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