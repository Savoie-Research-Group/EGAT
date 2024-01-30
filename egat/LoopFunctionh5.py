
from ast import excepthandler
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
from ast import excepthandler
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

from graphgenhelperfunctions import *



def LoopFunctionfromh5(hf,df,output_F,Rind,fail=os.path.join(os.getcwd(),'Failure.txt'),getlp=False,getradical=False,folder=None,addons = False):
    num2element = {1:'H', 6:'C', 7:'N', 8:'O'}
    element_encode = {'H': [1,1.00794],'C':[6,12.011],'N':[7,14.00674],'O':[8,15.9994]}
    bond_encode = {'T0': [0,0,1],'T1':[0,1,0],'T2':[1,0,0]}

    rxn = df.iloc[Rind]
    
    # initialize a empty list
    info = {}


    RxnDE   = rxn.DE_F
    RxnDH   = rxn.DH
    
    Rxn = hf[Rind]
    # initialize a empty list
    info = {}
    # parse elements
    elements = [num2element[Ei] for Ei in np.array(Rxn.get('elements'))]

    # parse geometries
    R_G  = np.array(Rxn.get('RG'))
    P_G  = np.array(Rxn.get('PG'))

    Rsmiles,Psmiles,am_Rsmiles,am_Psmiles = getSMILESfrom3D(elements,R_G,P_G,Rind)
    
    Rinchi = getInchifromSMILES(Rsmiles)
    Pinchi = getInchifromSMILES(Psmiles)
    

    # generate adj mat
    R_adj = Table_generator(elements,R_G,scale_factor=1.2)
    P_adj = Table_generator(elements,P_G,scale_factor=1.2)
    if R_adj is False or P_adj is False:
        with open(fail,'a') as fp:
            fp.write("Failure in in Table_generator when paring {}, skip...".format(Rind))
        return None

    # generate bond mat
    Rscores,R_bond_mats,_ = find_lewis(elements,R_adj)
    Pscores,P_bond_mats,_ = find_lewis(elements,P_adj)


    # parse Reaction type
    if folder == 'Molecularity':
        # For reaction type, get it from the column itself, if it doesn't exist get it from the Smiles strings themselves. 
        try: 
            Rxntype = 'R{}P{}'.format(rxn.NR,rxn.NP)
        except:
            Rxntype = 'R{}P{}'.format(len(Rsmiles.split('.')),len(Psmiles.split('.')))
    elif folder == 'Rtype':
        b,f = return_bnfn(elements,R_bond_mats,P_bond_mats)
        Rxntype = 'b{}f{}'.format(b,f)
    elif folder == 'HA':
        HAcount = getHA(Rsmiles)
        Rxntype = '{}atom'.format(rxn.NR,rxn.NP)
    else:
        Rxntype = 'All'

    if Rscores[0] > 5 or Pscores[0] > 5:
        print('Skip {} due to incorrect lewis strcuture...'.format(Rind))
        with open(fail,'a') as fp:
            fp.write('Skip {} due to incorrect lewis strcuture...'.format(Rind))
        return None
    
    R_bond_mat = R_bond_mats[0]
    P_bond_mat = P_bond_mats[0]
    
    # generate RP-adj that at least one side is connected
    Redges_u,Redges_v  = [],[]
    Pedges_u,Pedges_v  = [],[]
    for i in range(len(elements)):
        for j in range(len(elements)):
            if R_adj[i][j] > 0:
                Redges_u.append(i)
                Redges_v.append(j)

            if P_adj[i][j] > 0:
                Pedges_u.append(i)
                Pedges_v.append(j)
                
    # generate distance matrix
    R_gs = graph_seps(R_adj)
    R_gs[R_gs < 0] = 100
    P_gs = graph_seps(P_adj)
    P_gs[P_gs < 0] = 100

    # obtain ring atom list
    R_ring_atoms = return_rings(adjmat_to_adjlist(R_adj),max_size=20,remove_fused=True)
    P_ring_atoms = return_rings(adjmat_to_adjlist(P_adj),max_size=20,remove_fused=True)

    #Obtain Radical Electron Counts 
    if getradical is True:
        #Obtain list of radicals and diradicals via RDKit
        R_Radicals = return_radicals_RDKit(am_Rsmiles)
        P_Radicals = return_radicals_RDKit(am_Psmiles) 
    
    #Obtain Lone Pair Electron Counts 
    if getlp is True:
        #Obtain list of radicals and diradicals via RDKit
        R_lp = return_lonepairs_RDKit(am_Rsmiles)
        P_lp = return_lonepairs_RDKit(am_Psmiles) 

    # obtain reactive atoms and bonds
    bond_changes,reactive_atoms = return_reactive(elements,R_bond_mat,P_bond_mat)

    # compute atom features
    ### atom feature: atomic number, atomic mass, number of hydrogens, number of C, number of N, number of O, distance to reactive atom, ring_atom (0 refers to false, 1 refers to true) dim: 1*8
    Ratom_features,Patom_features = [],[]
    for ind in range(len(elements)):
        Etype = element_encode[elements[ind]]
        R_neighbors = [elements[counti] for counti,i in enumerate(R_adj[ind,:]) if i != 0]
        R_NE_count = [R_neighbors.count('H'),R_neighbors.count('C'),R_neighbors.count('N'),R_neighbors.count('O')]
        P_neighbors = [elements[counti] for counti,i in enumerate(P_adj[ind,:]) if i != 0]
        P_NE_count = [P_neighbors.count('H'),P_neighbors.count('C'),P_neighbors.count('N'),P_neighbors.count('O')]
        R_dis  = min([R_gs[ind][indr] for indr in reactive_atoms])
        P_dis  = min([P_gs[ind][indr] for indr in reactive_atoms])

        if getradical is True:
            R_rad = [R_Radicals[ind]]
            P_rad = [P_Radicals[ind]]
        

        if getlp is True:
            R_lonepair = [R_lp[ind]]
            P_lonepair = [P_lp[ind]]

        # obtain ring information
        if True in [ind in ra_list for ra_list in R_ring_atoms]: RinR = 1
        else: RinR = 0

        if True in [ind in ra_list for ra_list in P_ring_atoms]: PinR = 1
        else: PinR = 0
            
        atomfeature_R = Etype+R_NE_count+[R_dis,RinR]
        atomfeature_P = Etype+P_NE_count+[P_dis,PinR]
        if getradical is True:
            atomfeature_R += R_rad
            atomfeature_P += P_rad
        
        if getlp is True:
            atomfeature_R += R_lonepair
            atomfeature_P += P_lonepair
        
        Ratom_features.append(atomfeature_R)
        Patom_features.append(atomfeature_P)

    # compute bond features for R
    ### bond feature: bond order, bond type (T0: non-changed bond, T1: bond order change, T2: bond change [i.e., break or form]), in ring or not (0 and 1), dim: 1*5
    Rbond_features = []
    for ind in range(len(Redges_u)):

        # obtain atom pair for each "bond"
        edge_ind = sorted([Redges_u[ind],Redges_v[ind]])

        ### compute bond order
        BO_R = R_bond_mat[edge_ind[0],edge_ind[1]]
        BO_P = P_bond_mat[edge_ind[0],edge_ind[1]]

        ### analyze the bond type
        # bond type (T0: non-changed bond, T1: bond order change, T2: bond change [i.e., break or form])
        if BO_R == BO_P: Btype = 'T0'
        elif BO_P == 0.0: Btype = 'T2'
        else: Btype = 'T1'
        Btype = bond_encode[Btype]

        ### analyze ring information
        if True in [(edge_ind[0] in ra_list and edge_ind[1] in ra_list) for ra_list in R_ring_atoms]: RinR = 1
        else: RinP = 0

        bondfeature_R = [BO_R]+Btype+[RinR]
        Rbond_features.append(bondfeature_R)

    # compute bond features
    ### bond feature: bond order, bond type (T0: non-changed bond, T1: bond order change, T2: bond change [i.e., break or form]), in ring or not (0 and 1), dim: 1*5
    Pbond_features = []
    for ind in range(len(Pedges_u)):

        # obtain atom pair for each "bond"
        edge_ind = sorted([Pedges_u[ind],Pedges_v[ind]])

        ### compute bond order
        BO_R = R_bond_mat[edge_ind[0],edge_ind[1]]
        BO_P = P_bond_mat[edge_ind[0],edge_ind[1]]

        ### analyze the bond type
        # bond type (T0: non-changed bond, T1: bond order change, T2: bond change [i.e., break or form])
        if BO_R == BO_P: Btype = 'T0'
        elif BO_R == 0.0: Btype = 'T2'
        else: Btype = 'T1'
        Btype = bond_encode[Btype]

        ### analyze ring information
        if True in [(edge_ind[0] in ra_list and edge_ind[1] in ra_list) for ra_list in P_ring_atoms]: PinR = 1
        else: PinR = 0
            
        bondfeature_P = [BO_P]+Btype+[PinR]
        Pbond_features.append(bondfeature_P)


    # Get the Rsmiles and Psmiles from the Geometry


    # pack info into one list
    info['Ru'] = Redges_u
    info['Rv'] = Redges_v
    info['Pu'] = Pedges_u
    info['Pv'] = Pedges_v
    info['atom_F_R'] = Ratom_features
    info['atom_F_P'] = Patom_features
    info['bond_F_R'] = Rbond_features
    info['bond_F_P'] = Pbond_features
    
    info['Index'] = Rind
    info['Rsmiles'] = Rsmiles
    info['Psmiles'] = Psmiles
    info['Rinchi'] = Rinchi
    info['Pinchi'] = Pinchi
    
    info['DE']       = RxnDE
    info['rxnH']     = RxnDH
    
    with open('{}/{}/{}.json'.format(output_F,Rxntype,Rind), 'w') as filehandle:
        json.dump(info, filehandle)




