from ast import excepthandler
from copy import deepcopy
import sys
#from winreg import REG_NOTIFY_CHANGE_ATTRIBUTES
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
import omegaconf
from graphgenhelperfunctions import *
from openshellRDKit import *
from RDKitAtomMapping import GenerateAtomMapping
from RDKitSMILESGen import *
from RDkitHelpers import *
import sqlite3
import pymongo
import periodictable
''' 
SET UP THE NUMERICAL NOTATION FOR THE ATOM AND BOND VECTORS
### atom feature (from taffi): dim=1*9
# - atomic number
# - atomic mass
# - Number of hydrogens
# - Number of C
# - Number of N
# - Number of O
# - Distance to reactive atom
# - Is the atom in a ring (0 refers to false, 1 refers to true)
# - formal_charge
### atom feature (from rdkit): dim=1*8
# - chiral_tag (1*3, one hot)
# - hybridization (1*4, one hot)
# - aromaticity (0 refers to false, 1 refers to true)  
### atom feature (from EGAT,optional): dim=1*8
# - Spiro Atom (0 refers to false, 1 refers to true) 
# - Bridgehead Atom (0 refers to false, 1 refers to true)
# - Electronegativity
# - Hydrogen Bonding Information (0 refers to false, 1 refers to true for donor and acceptor)  
### Grambow's work: one-hot encoding of the atomic number, the degree, the formal charge, the chiral tag, the total number of hydrogens, and the hybridization; an aromaticity flag; the atomic mass; in ring or not

### bond feature (from taffi): dim= 1*5
# - bond type (T1: non-changed bond, T2: bond order goes up, T3: bond break, T4: bond form,T5: bond order goes down)
#    TO-DO: Change T2 to bond order up and add T5 as bond order down. 
# - in ring or not (0 and 1)
### bond feature (from rdkit): dim=1*9 (three 1*3 one-hot vectors 1*9)
#  - bond type (B1: 0, B2-B4: 1-3, B5: aromaticity)
#  - Conjugation
#  - Bond stereo 
### bond feature (based from mordred,optional): dim=1*2
# - Bond Rotatability 
# - Bond Polerity
### Grambow's work: whether the bond is a single, double, triple, or aromatic bond; whether it is conjugated; whether it is in a ring; one-hot encoding of the bond stereochemistry

'''


'''
gethbondinfo=False,,getabinfo=False
'''

def loopfunctionfromcsv(Rind,data_path,input,target,addtional,smiles,molecular=False,method_mapping='RxnMapper',folder=None,getradical=None,onlyH=False,getbondrot=False,atom_map=False,removeelementinfo=False,removeneighborcount=False,removeringinfo=False,removereactiveinfo=False,
removeformalchargeinfo=False,removearomaticityinfo=False,removechiralinfo=False,removehybridinfo=False,removebondtypeinfo=False,
removebondorderinfo=False,removeconjinfo=False,removestereoinfo=False,getRDKitFeatures=False,getRDKITNormfatures=False,datastorage=None,isOldEdition=False,useFullHyb=False,geteneg=False,getbpol=False,getspiroinfo=False,getbridgeinfo=False,explicit_H=False,num_workers=1):
        ###### READ CSV FILE 
        if input[-3:] == 'csv':
            df = pd.read_csv(input,index_col = 0)
        elif input[-3:] == 'tsv':
            df = pd.read_csv(input,sep='\t')
        pt = Chem.GetPeriodicTable()
        #Future Work: Expand num2element and element_encode to include all atoms. Expand Hybridization to include the other nodes too. 
        num2element = {1:'H', 6:'C', 7:'N', 8:'O',pt.GetAtomicNumber('P'):'P',pt.GetAtomicNumber('S'):'S',pt.GetAtomicNumber('F'):'F',pt.GetAtomicNumber('Cl'):'Cl',pt.GetAtomicNumber('Br'):'Br',pt.GetAtomicNumber('I'):'I'}
        if isOldEdition:
            element_encode = {'H': [1,1.00794],'C':[6,12.011],'N':[7,14.00674],'O':[8,15.9994],'F':[pt.GetAtomicNumber('F'),pt.GetAtomicWeight(pt.GetAtomicNumber('F'))],
            'Cl':[pt.GetAtomicNumber('Cl'),pt.GetAtomicWeight(pt.GetAtomicNumber('Cl'))],'Br':[pt.GetAtomicNumber('Br'),pt.GetAtomicWeight(pt.GetAtomicNumber('Br'))],
            'S':[pt.GetAtomicNumber('S'),pt.GetAtomicWeight(pt.GetAtomicNumber('S'))],'P':[pt.GetAtomicNumber('P'),pt.GetAtomicWeight(pt.GetAtomicNumber('P'))],'I':[pt.GetAtomicNumber('I'),pt.GetAtomicWeight(pt.GetAtomicNumber('I'))]}

        else:
            all_elements = periodictable.elements
            element_encode = dict()
            for i,elements in enumerate(all_elements):
                if elements.symbol != 'n':
                    element_encode[elements.symbol] = [elements.number,elements.mass]
                
      
        
        atom_chiral_encode = {'?': [0,0,1], 'R': [0,1,0], 'S': [1,0,0]}
        if not useFullHyb: atom_hybrid_encode = {Chem.HybridizationType.S: [0,0,0,1], Chem.HybridizationType.SP: [0,0,1,0], Chem.HybridizationType.SP2: [0,1,0,0], Chem.HybridizationType.SP3: [1,0,0,0]}
        else: atom_hybrid_encode = {Chem.HybridizationType.S: [0,0,0,1,0,0,0,0,0], Chem.HybridizationType.SP: [0,0,1,0,0,0,0,0,0], Chem.HybridizationType.SP2: [0,1,0,0,0,0,0,0,0], Chem.HybridizationType.SP3: [1,0,0,0,0,0,0,0,0],Chem.HybridizationType.SP2D: [0,0,0,0,1,0,0,0,0],Chem.HybridizationType.SP3D: [0,0,0,0,0,1,0,0,0],Chem.HybridizationType.SP3D2: [0,0,0,0,0,0,1,0,0],Chem.HybridizationType.OTHER: [0,0,0,0,0,0,0,1,0],Chem.HybridizationType.UNSPECIFIED: [0,0,0,0,0,0,0,0,1]}
        if not isOldEdition: bond_encode = {'T1': [0,0,0,1,0],'T2':[0,0,1,0,0],'T3':[0,1,0,0,0],'T4':[1,0,0,0,0],'T5':[0,0,0,0,1]}
        else: bond_encode = {'T1': [0,0,0,1],'T2':[0,0,1,0],'T3':[0,1,0,0],'T4':[1,0,0,0]}
        bond_order_encode = {'B0': [0,0,0,0,1],'B1':[0,0,0,1,0],'B2':[0,0,1,0,0],'B3':[0,1,0,0,0],'BA':[1,0,0,0,0]}
        bond_stereo_encode = {'ANY': [0,0,1], 'E': [0,1,0], 'Z': [1,0,0]}
        bond_rotat_encode = {'TRUE':[1,0],'FALSE':[0,1]}
        try:
            ###### PARSE THE ELEMENT IN THE DATAFRAME AND SET UP THE LIST WE SAVE THE DATA TO
            rxn = df.iloc[Rind]
            info = {}
            info['Indices'] = Rind
            ###### SAVE THE TARGET AND ADDTIONAL DATA TO THE DICTIONARY
            if isinstance(target,list) or isinstance(target,omegaconf.listconfig.ListConfig):
                for outputs in target:
                    info[outputs] = rxn[outputs]
            else:
                info[target] = rxn[target]
            adddict = dict()
            if isinstance(addtional,list)or isinstance(addtional,omegaconf.listconfig.ListConfig):
                for outputs in addtional:
                    info[outputs] = rxn[outputs]
            elif addtional is not None:
                info[addtional] = rxn[addtional]
            ###### GET THE SMILES STRINGS AND SAVE THE inchi keys TO THE DICTIONARY
            if not molecular:
                smiles = rxn[smiles].split('>>')
                Rsmiles = smiles[0]
                Psmiles = smiles[1]        
                info['Rinchi'] = getInchifromSMILES(Rsmiles)
                info['Pinchi'] = getInchifromSMILES(Psmiles)
            else:
                Rsmiles = rxn[smiles]
                info['Rinchi'] = getInchifromSMILES(Rsmiles)
            ###### IF THEY ARE NOT ATOM-MAPPED, ATOM-MAP THEM USING THE HELPER FUNCTION WE WROTE
            # Note: Right now, we can use RxnMapper as a stop-gap to get some kind of working solution. 
            # One way to manually do it is by aligning the R and P geometries we generate from openbabel and then atom-map w.r.t to RMSD.
            if not molecular: 
                if atom_map is True:
                    NRsmiles,NPsmiles = GenerateAtomMapping(Rsmiles,Psmiles,method_mapping)
                    info["Rsmiles"] = Rsmiles
                    info["Psmiles"] = Psmiles
                    Rsmiles = NRsmiles
                    Psmiles = NPsmiles
                else:
                    NRsmiles = RemoveMapping(Rsmiles)
                    NPsmiles = RemoveMapping(Psmiles)  
                    info["Rsmiles"] = Chem.MolToSmiles(NRsmiles)
                    info["Psmiles"] = Chem.MolToSmiles(NPsmiles) 
            else:
                if atom_map is True:
                    NRsmiles = deepcopy(Rsmiles)
                    Rsmiles = Chem.MolFromSmiles(NRsmiles)
                    if not explicit_H:
                        Rsmiles = Chem.AddHs(Rsmiles)
                    
                    Rsmiles = Chem.MolToSmiles(mol_with_atom_index(Rsmiles))
                    info["Rsmiles"] = NRsmiles
                else:
                    NRsmiles = RemoveMapping(Rsmiles)
                    info["Rsmiles"] = NRsmiles
            ###### PARSE ELEMENTS AND GET THE RELEVANT MATRICES 
            E_smiles_R, R_adj, R_bond_mat, R_fc = return_matrix(Rsmiles)
            if not molecular:
                E_smiles_P, P_adj, P_bond_mat, P_fc = return_matrix(Psmiles)
            ###### CHECK THE CONSISTENCY OF ELEMENTS, IF THEY ARE NOT CONSISTENT, EXCLUDE THEM FROM THE GRAPH GENERATION.
            # Note for the future: We might need to revisit this and try to see if we can 'balance' them in some way.  
            if not molecular:
                if E_smiles_R != E_smiles_P:
                    with open(os.path.join(data_path,'exclude.txt'),'a') as f:
                        f.write('{}\n'.format(Rind))
                        f.write(f'The reaction between {Rsmiles} and {Psmiles} failed because the element matrix is imbalnaced. \n')
                        print(f'The reaction between {Rsmiles} and {Psmiles} failed because the element matrix is imbalnaced. \n')
                    return None 
                else:
                    elements = E_smiles_R
            else:
                elements = E_smiles_R
            ###### MAKE THE JSON FOLDERS THAT STORE THE VECTOR INFORMATION BY A CERTAIN TYPE.
            # There are four ways: Molecularity, Reaction Type (See the YARP paper for that), Heavy Atom Count, or just dump them all in one place
            if folder == 'Molecularity':
                if not molecular:
                    Rxntype = 'R{}P{}'.format(len(Rsmiles.split('.')),len(Psmiles.split('.')))
                else:
                    Rxntype = 'R{}'.format(len(Rsmiles.split('.')))
            elif folder == 'Rtype':
                if not molecular:
                    B,F = return_bnfn(elements,R_bond_mat,P_bond_mat)
                    Rxntype = 'b{}f{}'.format(B,F)
                else:
                    return None
            elif folder == 'HA':
                HAcount = getHA(Rsmiles)
                Rxntype = '{}atom'.format(HAcount)
            else:
                Rxntype = 'All'
            

            if datastorage is None:
                if os.path.isdir(os.path.join(data_path,Rxntype)) is False:
                    os.mkdir(os.path.join(data_path,Rxntype))
                    # Write the folder into a .txt file 
                    with open(os.path.join(data_path,'Rxntype.txt'),'a') as f:
                        f.write('{}\n'.format(Rxntype))
            else:
                if os.path.isfile(os.path.join(data_path,'Rxntype.txt')):
                    with open(os.path.join(data_path,'Rxntype.txt'),'r') as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines]
                    if Rxntype not in lines:
                        with open(os.path.join(data_path,'Rxntype.txt'),'a') as f:
                            f.write('{}\n'.format(Rxntype))
                else:
                    with open(os.path.join(data_path,'Rxntype.txt'),'a') as f:
                        f.write('{}\n'.format(Rxntype))

            
            ###### GENERATE THE RP-ADJACENCY MATRIX SO THAT AT LEAST ONE SIDE IS CONNECTED 
            edges_u,edges_v  = [],[]
            for i in range(len(elements)):
                for j in range(len(elements)):
                    if not molecular:
                        if R_adj[i][j] > 0 or P_adj[i][j] > 0:
                            edges_u.append(i)
                            edges_v.append(j)
                    else:
                        if R_adj[i][j] > 0:
                            edges_u.append(i)
                            edges_v.append(j)
            
            ###### GENERATE THE DISTANCE MATRIX
            R_gs = graph_seps(R_adj)
            R_gs[R_gs < 0] = 100
            if not molecular:
                P_gs = graph_seps(P_adj)
                P_gs[P_gs < 0] = 100
            ###### GET THE LIST OF RING ATOMS 
            R_ring_atoms = return_rings(adjmat_to_adjlist(R_adj),max_size=20,remove_fused=True)
            if not molecular:
                P_ring_atoms = return_rings(adjmat_to_adjlist(P_adj),max_size=20,remove_fused=True)
            ###### GET THE REACTIVE ATOMS AND BONDS 
            if not molecular: 
                bond_changes,reactive_atoms,bond_formed,bond_broken,bond_ochangeup,bond_ochangedown = return_reactive(elements,R_bond_mat,P_bond_mat)
            ###### GET THE LOCATION OF RADICALS
            if getradical == 'RDKit':
                #Obtain list of radicals and diradicals via RDKit
                R_Radicals = return_radicals_RDKit(Rsmiles)
                R_lp = return_lonepairs_RDKit(Rsmiles)
                if not molecular:
                    P_Radicals = return_radicals_RDKit(Psmiles) 
                    P_lp = return_lonepairs_RDKit(Psmiles) 

            elif getradical == 'YARP':
                R_lp,R_Radicals = getradicalsYARP(Rsmiles)
                if not molecular:
                    P_lp,P_Radicals = getradicalsYARP(Psmiles)
        
            ###### GET THE LOCATION OF ROTATABLE BONDS
            if getbondrot is True:
                R_br = getRotatableBondCount(Rsmiles)
                if not molecular:
                    P_br = getRotatableBondCount(Psmiles)
            
            if geteneg:
                R_eneg = get_electronegativity(Rsmiles)
                if not molecular:
                    P_eneg = get_electronegativity(Psmiles)
            
            if getspiroinfo:
                R_spiro = GetSpiroAtoms(Rsmiles)
                if not molecular:
                    P_spiro = get_electronegativity(Psmiles)
            
            if getbridgeinfo:
                R_brid = GetBridgeheadAtoms(Rsmiles)
                if not molecular:
                    P_brid = GetBridgeheadAtoms(Psmiles)
            
            if getbpol:
                R_bpol = calculate_bond_polarity(Rsmiles)
                if not molecular:
                    P_bpol = calculate_bond_polarity(Rsmiles)

            ###### GET THE BOND STEREOCHEMISTRY, ATOM HYBRIDIZATION AND AROMATICITY, LOCATION OF CHIRAL CENTERS. 
            R_CT,R_AA,R_HY,R_BS,R_Conj,R_BA = find_stereochemistry(Rsmiles)
            if not molecular:
                P_CT,P_AA,P_HY,P_BS,P_Conj,P_BA = find_stereochemistry(Psmiles)
            
            

            try:
                # Save the features 
                if not molecular:
                    Ratom_features,Patom_features = [],[]
                else:
                    Ratom_features = []

                for ind in range(len(elements)):
                    ###### GET ELEMENTS 
                    molecule = Chem.MolFromSmiles(Rsmiles)
                    atom_mappings = min([atom.GetAtomMapNum() for atom in molecule.GetAtoms()])
                    if atom_mappings == 0:
                        ind_in_mol = ind
                    else:
                        ind_in_mol = ind+1
                    Etype = element_encode[elements[ind]]
                    ###### GET NEIGHBORS AND THEIR TYPES
                    # There is an option to only look at Hydrogen Bonding
                    R_neighbors = [elements[counti] for counti,i in enumerate(R_adj[ind,:]) if i != 0]                
                    if onlyH:
                        R_NE_count = [R_neighbors.count('H')]
                    else:
                        R_NE_count = [R_neighbors.count('H'),R_neighbors.count('C'),R_neighbors.count('N'),R_neighbors.count('O')]

                    if not molecular:
                        P_neighbors = [elements[counti] for counti,i in enumerate(P_adj[ind,:]) if i != 0]
                        if onlyH:
                            P_NE_count = [P_neighbors.count('H')]
                        else:
                            P_NE_count = [P_neighbors.count('H'),P_neighbors.count('C'),P_neighbors.count('N'),P_neighbors.count('O')]

                    ###### GET REACTIVE ATOMS
                    if not molecular:
                        R_dis  = min([R_gs[ind][indr] for indr in reactive_atoms])
                        P_dis  = min([P_gs[ind][indr] for indr in reactive_atoms])

                    ###### CHECK IF ATOM IN A RING
                    if True in [ind in ra_list for ra_list in R_ring_atoms]: RinR = 1
                    else: RinR = 0
                    
                    if not molecular:
                        if True in [ind in ra_list for ra_list in P_ring_atoms]: PinR = 1
                        else: PinR = 0
                    
                    ###### GET AROMATICITY
                    if elements[ind] == 'H': Raromaticity = 0
                    elif R_AA[ind_in_mol]: Raromaticity = 1
                    else: Raromaticity = 0

                    if not molecular:
                        if elements[ind] == 'H': Paromaticity = 0
                        elif P_AA[ind_in_mol]: Paromaticity = 1
                        else: Paromaticity = 0

                    ###### GET HYBRIDIZATION
                    if elements[ind] == 'H': 
                        R_hybrid = [0,0,0,1]
                        if not molecular:
                            P_hybrid = [0,0,0,1]
                    else: 
                        R_hybrid = atom_hybrid_encode[R_HY[ind_in_mol]]
                        if not molecular:
                            P_hybrid = atom_hybrid_encode[P_HY[ind_in_mol]]

                    ###### CHECK IF IT IS IN A CHIRAL CENTER
                    if ind_in_mol in R_CT: R_chiral = atom_chiral_encode[R_CT[ind_in_mol]]
                    else: R_chiral = [0,0,1]

                    if not molecular:
                        if ind_in_mol in P_CT: P_chiral = atom_chiral_encode[P_CT[ind_in_mol]]
                        else: P_chiral = [0,0,1]

                    ###### MERGE ALL FEATURES 
                    # Note: If statements are used to check for feature importance (that is a later step of analysis) and for adding additional features 
                    atomfeature_P,atomfeature_R = [],[]
                    node_feats = 0
                    # Add Element info if not stated that you want to remove it from training. 
                    if not removeelementinfo:
                        atomfeature_R += Etype
                        node_feats += len(Etype)
                        if not molecular: atomfeature_P += Etype

                    # Add Neighbor info if not stated that you want to remove it from training. 
                    if not removeneighborcount:
                        atomfeature_R += R_NE_count
                        node_feats += len(R_NE_count)
                        if not molecular: atomfeature_P += P_NE_count

                    # Add reactive info if not stated that you want to remove it from training. 
                    if not removereactiveinfo:
                        if not molecular:
                            atomfeature_R += [R_dis]
                            node_feats += 1
                            atomfeature_P += [P_dis]
                    
                    # Add ring info if not stated that you want to remove it from training. 
                    if not removeringinfo:
                        atomfeature_R += [RinR]
                        node_feats += 1
                        if not molecular: atomfeature_P += [PinR]
                    
                    # Add charge info if not stated that you want to remove it from training. 
                    if not removeformalchargeinfo:
                        atomfeature_R += [R_fc[ind]]
                        node_feats += 1
                        if not molecular: atomfeature_P += [P_fc[ind]]

                    # Add aromaticity info if not stated that you want to remove it from training. 
                    if not removearomaticityinfo:
                        atomfeature_R += [Raromaticity]
                        node_feats += 1
                        if not molecular: atomfeature_P += [Paromaticity]
                    
                    # Add chiral info if not stated that you want to remove it from training. 
                    if not removechiralinfo:
                        atomfeature_R += R_chiral
                        node_feats += len(R_chiral)
                        if not molecular: atomfeature_P += P_chiral

                    # Add hybrid info if not stated that you want to remove it from training. 
                    if not removehybridinfo:
                        atomfeature_R += R_hybrid
                        node_feats += len(R_hybrid)
                        if not molecular: atomfeature_P += P_hybrid
                    
                    # Add radical info if stated. 
                    if getradical:
                        atomfeature_R += [R_Radicals[ind]]
                        node_feats += 1
                        if not molecular: atomfeature_P += [P_Radicals[ind]]
                        # Add lone pair info if stated.
                        atomfeature_R += [R_lp[ind]]
                        node_feats += 1
                        if not molecular: atomfeature_P += [P_lp[ind]]

                    
                    Ratom_features.append(atomfeature_R)
                    if not molecular: Patom_features.append(atomfeature_P)

                ###### GET BOND FEATURES 
                if not molecular:
                    Rbond_features,Pbond_features = [],[]
                else:
                    Rbond_features = []
                
                for ind in range(len(edges_u)):

                    ###### GET THE ATOM PAIRS FOR EACH BOND
                    edge_ind = sorted([edges_u[ind],edges_v[ind]])
                    edge_ind_mol = sorted([edges_u[ind]+1,edges_v[ind]+1])

                    ###### GET THE BOND ORDER
                    BO_R = R_bond_mat[edge_ind[0],edge_ind[1]]
                    if not molecular:
                        BO_P = P_bond_mat[edge_ind[0],edge_ind[1]]

                    ###### GET THE BOND TYPE
                    if not molecular:
                        if BO_R == BO_P: 
                            RBtype = 'T1'
                            PBtype = 'T1'
                        elif BO_R == 0.0:
                            RBtype = 'T4'
                            PBtype = 'T3'
                        elif BO_P == 0.0: 
                            RBtype = 'T3'
                            PBtype = 'T4'
                        elif BO_R < BO_P and BO_R > 0:
                            RBtype = 'T2'
                            PBtype = 'T2'
                        elif BO_R > BO_P and BO_R > 0:
                            RBtype = 'T5'
                            PBtype = 'T5'
                        
                        RBtype = bond_encode[RBtype]
                        PBtype = bond_encode[PBtype]



                    ###### CHECK IF BOND IS IN A RING
                    if True in [(edge_ind[0] in ra_list and edge_ind[1] in ra_list) for ra_list in R_ring_atoms]: RinR = 1
                    else: RinR = 0

                    if not molecular:
                        if True in [(edge_ind[0] in ra_list and edge_ind[1] in ra_list) for ra_list in P_ring_atoms]: PinR = 1
                        else: PinR = 0
                        

                    ###### MERGE RDKIT FEATURES
                    bondfeature_R = []
                    bondfeature_P = []
                    edge_feats = 0
                    if not molecular:
                        if not removebondtypeinfo:
                            bondfeature_R += RBtype
                            bondfeature_P += PBtype
                            edge_feats += len(RBtype)
                    
                    if not molecular:
                        if not removeringinfo:
                            bondfeature_R += [RinR]
                            bondfeature_P += [PinR]
                            edge_feats += 1
                    else:
                        if not removeringinfo:
                            bondfeature_R += [RinR]
                            edge_feats += 1
                    
                    # check if a bond is in reactant bond feature dict or not
                    ###### CHECK IF A BOND IS IN REACTANT BOND FEATURE DICT OR NOT 
                    if BO_R == 0:

                        if not removebondorderinfo:
                            bondfeature_R += [0,0,0,0,1]
                            edge_feats += 5
                        # Add Bond Conjugation info if not stated that you want to remove it from training. 
                        if not removeconjinfo:
                            bondfeature_R += [0]
                            edge_feats += 1
                        
                        # Add Bond Stereochemistry info if not stated that you want to remove it from training. 
                        if not removestereoinfo:
                            bondfeature_R += [0,0,0]
                            edge_feats += 3

                        if getbondrot:
                            bondfeature_R += [0,0]
                            edge_feats += 2

                    else:
                        # parse bond order type
                        if tuple(edge_ind_mol) in R_BA and R_BA[tuple(edge_ind_mol)]: R_bond_type = bond_order_encode['BA']
                        else: R_bond_type = bond_order_encode['B{}'.format(int(BO_R))]

                        ###### CHECK BOND CONJUGATION
                        if tuple(edge_ind_mol) in R_Conj and R_Conj[tuple(edge_ind_mol)]: R_bond_conj = [1]
                        else: R_bond_conj = [0]
                    
                        ###### CHECK BOND STEREOCHEM
                        if tuple(edge_ind_mol) in R_BS: R_bond_stereo = bond_stereo_encode[R_BS[tuple(edge_ind_mol)]]
                        else: R_bond_stereo = [0,0,1]
                
                        if not removebondorderinfo:
                            bondfeature_R += R_bond_type
                            edge_feats += len(R_bond_type)
                        # Add Bond Conjugation info if not stated that you want to remove it from training. 
                        if not removeconjinfo:
                            bondfeature_R += R_bond_conj
                            edge_feats += len(R_bond_conj)
                        
                        # Add Bond Stereochemistry info if not stated that you want to remove it from training. 
                        if not removestereoinfo:
                            bondfeature_R += R_bond_stereo
                            edge_feats += len(R_bond_stereo)
        
                        if getbondrot:
                            if tuple(edge_ind_mol) in R_br:
                                bondfeature_R += bond_rotat_encode['TRUE']
                            else:
                                bondfeature_R += bond_rotat_encode['FALSE']
                            
                            edge_feats += len(bond_rotat_encode['TRUE'])
                        
                    # check if a bond is in reactant bond feature dict or not
                    if not molecular:
                        if BO_P == 0:

                            # Add Bond Order info if not stated that you want to remove it from training. 
                            if not removebondorderinfo:
                                bondfeature_P += [0,0,0,0,1]
                        
                            # Add Bond Conjugation info if not stated that you want to remove it from training. 
                            if not removeconjinfo:
                                bondfeature_P += [0]
                            
                            # Add Bond Stereochemistry info if not stated that you want to remove it from training. 
                            if not removestereoinfo:
                                bondfeature_P += [0,0,0]
                                
                            
                            if getbondrot:
                                bondfeature_P += [0,0]
                        else:
                            
                            if tuple(edge_ind_mol) in P_BA and P_BA[tuple(edge_ind_mol)]: P_bond_type = bond_order_encode['BA']
                            else: P_bond_type = bond_order_encode['B{}'.format(int(BO_R))]

                            # parse conjugation
                            if tuple(edge_ind_mol) in P_Conj and P_Conj[tuple(edge_ind_mol)]: P_bond_conj = [1]
                            else: P_bond_conj = [0]
                        
                            # parse stereochemistry
                            if tuple(edge_ind_mol) in P_BS: P_bond_stereo = bond_stereo_encode[P_BS[tuple(edge_ind_mol)]]
                            else: P_bond_stereo = [0,0,1]
                            
                            # merge into bond feature
                            # Add Bond Order info if not stated that you want to remove it from training. 
                            if not removebondorderinfo:
                                bondfeature_P += P_bond_type
                            
                            # Add Bond Conjugation info if not stated that you want to remove it from training. 
                            if not removeconjinfo:
                                bondfeature_P += P_bond_conj
                            
                            # Add Bond Stereochemistry info if not stated that you want to remove it from training. 
                            if not removestereoinfo:
                                bondfeature_P += P_bond_stereo
        
                            if getbondrot:
                                if tuple(edge_ind_mol) in P_br:
                                    bondfeature_R += bond_rotat_encode['TRUE']
                                else:
                                    bondfeature_R += bond_rotat_encode['FALSE']

                    Rbond_features.append(bondfeature_R)
                    if not molecular:
                        Pbond_features.append(bondfeature_P)

                #print(node_feats,edge_feats)
                ###### ADD THE GLOBAL FEATURES USING DESCRIPTASTORUS
                if getRDKitFeatures or getRDKITNormfatures:
                    if getRDKitFeatures:
                        generator = rdDescriptors.RDKit2D()
                    else:
                        generator = rdNormalizedDescriptors.RDKit2DNormalized()
                    if not molecular:
                        smilist = [Rsmiles,Psmiles]
                    else:
                        smilist = [Rsmiles]
                    results = generator.process(smilist)
                    if results[0] is None:
                        with open(os.path.join(data_path,'exclude.txt'),'a') as f:
                            f.write('{}\n'.format(Rind))
                            f.write(f'The reaction between {Rsmiles} and {Psmiles} failed because the Additional Descriptor failed. \n')
                        return None 
                    elif results[0] is False:
                        with open(os.path.join(data_path,'exclude.txt'),'a') as f:
                            f.write('{}\n'.format(Rind))
                            f.write(f'The reaction between {Rsmiles} and {Psmiles} failed because the Additional Descriptor failed. \n')
                        return None 
                    else:
                        info['R_Addon'] = list(results[1])
                        if not molecular:
                            info['P_Addon'] = list(results[2])

                # pack info into one list
                info['u'] = edges_u
                info['v'] = edges_v
                info['atom_F_R'] = Ratom_features
                info['bond_F_R'] = Rbond_features
                if not molecular:
                    info['atom_F_P'] = Patom_features
                    info['bond_F_P'] = Pbond_features

                if datastorage is None:
                    with open('{}/{}/{}.json'.format(data_path,Rxntype,Rind), 'w') as filehandle:
                        json.dump(info, filehandle)
                elif datastorage == 'SQL':
                    if num_workers == 1:
                        database_path = data_path.split('/')[-1]+'.db'
                        database_path = os.path.join(data_path,database_path)
                        connection = sqlite3.connect(database_path)
                        cursor = connection.cursor()
                        info['rxntype'] = Rxntype
                        if 'split' in rxn.columns:
                            info['split'] = rxn['split']
                        



                        table_name = data_path.split('/')[-1] + '_' + Rxntype + '_' + str(Rind)

                        create_table_query = f"""CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, value TEXT)"""
                        cursor.execute(create_table_query)
                        for key,value in info.items():
                            insert_query = f"INSERT INTO {table_name} (key,value) VALUES (?,?)"
                            cursor.execute(insert_query,(str(key),str(value)))
                        

                        connection.commit()
                        connection.close()
                    else:
                        info['rxntype'] = Rxntype
                    


                #print('done')
            except Exception as e:
                print(e)
                with open(os.path.join(data_path,'fail.txt'),'a') as ff:
                    ff.write('{}\n'.format(Rind))
                    if not molecular:
                        ff.write(f'The reaction between {Rsmiles} and {Psmiles} failed because of {e} \n')
                    else:
                        ff.write(f'The reaction between {Rsmiles} failed because of {e} \n')
                import traceback
                traceback.print_exc()
                pass
        except Exception as e:
            with open(os.path.join(data_path,'fail.txt'),'a') as ff:
                    ff.write('{}\n'.format(Rind))
                    if not molecular:
                        ff.write(f'The reaction between {Rsmiles} and {Psmiles} failed because of {e} \n')
                    else:
                        ff.write(f'The reaction between {Rsmiles} failed because of {e} \n')
                        print(f'The reaction between {Rsmiles} failed because of {e} \n')
            
            pass
        
        if datastorage == 'SQL' and num_workers != 1:
            #print('Done')
            return info
        else:
            return None