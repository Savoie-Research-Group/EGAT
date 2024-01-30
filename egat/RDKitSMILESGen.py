from ast import excepthandler
import sys
import h5py
sys.path.append('../utilities')
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


# Function that takes a 3D geometry and gets the SMILES 
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 np.array holding the geometry of the molecule
#               Mode:  Optional. The way you want to find smiles. 
# Returns:      SMILES of reactant and product
def getSMILESfrom3DRDKit(E,RG,PG,Rind,mode='TAFFI'):

    # generate adj_mats
    Radj_mat = Table_generator(E,RG)
    Padj_mat = Table_generator(E,PG)

    # generate am-smiles
    # Attempt 1: generate mol and smiles by taffi
    if mode == 'TAFFI':
        try:
            mol_write("{}_R.mol".format(Rind),E,RG,Radj_mat)
            am_Rsmiles = return_atommaped_smi("{}_R.mol".format(Rind))
            substring = "obabel -imol {}_R.mol -ocan".format(Rind)
            output    = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            NRsmiles  = output.split()[0]
        except:
             NRsmiles = 'Nonexistent'
             am_Rsmiles = 'Nonexistent'

        try:
            mol_write("{}_P.mol".format(Rind),E,PG,Padj_mat)
            am_Psmiles = return_atommaped_smi("{}_R.mol".format(Rind))
            substring = "obabel -imol {}_P.mol -ocan".format(Rind)
            output    = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            NPsmiles  = output.split()[0]
        except:
             NPsmiles = 'Nonexistent'
             am_Psmiles = 'Nonexistent'

    elif mode == 'xyz2mol':
        xyz_write("{}_R.xyz".format(Rind),E,RG)
        substring = "python utilities/xyz2mol/xyz2mol.py {}_R.xyz -o smiles --use-huckel".format(Rind)
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')    
        if len(output) > 0:
            NRsmiles = output.split()[0]
            substring = "/home/zhao922/bin/Github_public/xyz2mol/env/bin/python /home/zhao922/bin/Github_public/xyz2mol/xyz2mol.py {}_R.xyz -o sdf --use-huckel".format(Rind)
            output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            with open("{}_R.mol".format(Rind),'w') as f: f.write(output)
            am_Rsmiles = return_atommaped_smi("{}_R.mol".format(Rind))
        else:
            NRsmiles = 'Nonexistent'
            am_Rsmiles = 'Nonexistent'


        xyz_write("{}_P.xyz".format(Rind),E,PG)
        substring = "python utilities/xyz2mol/xyz2mol.py {}_P.xyz -o smiles --use-huckel".format(Rind)
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')    
        if len(output) > 0:
            NPsmiles = output.split()[0]
            substring = "python utilities/xyz2mol/xyz2mol.py {}_P.xyz -o sdf --use-huckel".format(Rind)
            output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            with open("{}_R.mol".format(Rind),'w') as f: f.write(output)
            am_Psmiles = return_atommaped_smi("{}_P.mol".format(Rind))
        else:
            NPsmiles = 'Nonexistent'
            am_Psmiles = 'Nonexistent'
    else:
        try:
            substring = "obabel -ixyz {}_R.xyz -omol".format(Rind)
            output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            with open("{}_R.mol".format(Rind),'w') as f: f.write(output)
            NRsmiles = return_smi("{}_R.mol".format(Rind))
            am_Rsmiles = return_atommaped_smi("{}_R.mol".format(Rind))
            os.remove("{}_R.mol".format(Rind))
            os.remove("{}_R.xyz".format(Rind))
        except:
            NRsmiles = 'Nonexistent'
            am_Rsmiles = 'Nonexistent'


        try:
            substring = "obabel -ixyz {}_P.xyz -omol".format(Rind)
            output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            with open("{}_P.mol".format(Rind),'w') as f: f.write(output)
            NRsmiles = return_smi("{}_P.mol".format(Rind))
            am_Rsmiles = return_atommaped_smi("{}_R.mol".format(Rind))
            os.remove("{}_P.mol".format(Rind))
            os.remove("{}_P.xyz".format(Rind))
        except:
            NPsmiles = 'Nonexistent'
            am_Psmiles = 'Nonexistent'
    
    return NRsmiles,NPsmiles,am_Rsmiles,am_Psmiles
