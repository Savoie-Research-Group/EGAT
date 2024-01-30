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


# Generates the adjacency matrix based on UFF bond radii
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 np.array holding the geometry of the molecule
#               File:  Optional. If Table_generator encounters a problem then it is often useful to have the name of the file the geometry came from printed. 
# Returns:      Adj_mat: Adjacency Matrix displaying the connectivity of the molecule
def Table_generator(Elements,Geometry,scale_factor=1.2):

    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # NOTE: Units of angstroms 
    # NOTE: These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    Radii = {  'H':0.354, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.244, 'Si':1.117,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # SAME AS ABOVE BUT WITH A SMALLER VALUE FOR THE Al RADIUS ( I think that it tends to predict a bond where none are expected
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.400, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # Use Radii json file in Lib folder if sepcified
    Max_Bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                   'K':None, 'Ca':None, 'Sc':15, 'Ti':14,  'V':13, 'Cr':12, 'Mn':11, 'Fe':10, 'Co':9, 'Ni':8, 'Cu':None, 'Zn':None, 'Ga':3,    'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':15, 'Zr':14, 'Nb':13, 'Mo':12, 'Tc':11, 'Ru':10, 'Rh':9, 'Pd':8, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':15, 'Hf':14, 'Ta':13,  'W':12, 'Re':11, 'Os':10, 'Ir':9, 'Pt':8, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
                     
    # Scale factor is used for determining the bonding threshold. 1.2 is a heuristic that give some lattitude in defining bonds since the UFF radii correspond to equilibrium lengths. 

    # Print warning for uncoded elements.
    for i in Elements:
        if i not in Radii.keys():
            print( "ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i)+\
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(Geometry,Geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Initialize Adjacency Matrix
    Adj_mat = np.zeros([len(Geometry),len(Geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*scale_factor:            
            Adj_mat[i,y_ind[count]]=1
    
        if Elements[i] == 'H' and Elements[y_ind[count]] == 'H':
            if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*1.5:
                Adj_mat[i,y_ind[count]]=1

    # Hermitize Adj_mat
    Adj_mat=Adj_mat + Adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in Radii.keys() }
    conditions = { "H":1, "C":4, "F":1, "Cl":1, "Br":1, "I":1, "O":2, "N":4, "B":4 }
    for count_i,i in enumerate(Adj_mat):

        if Max_Bonds[Elements[count_i]] is not None and sum(i) > Max_Bonds[Elements[count_i]]:
            problem_dict[Elements[count_i]] += 1
            cons = sorted([ (Dist_Mat[count_i,count_j],count_j) if count_j > count_i else (Dist_Mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(Adj_mat[count_i]) > Max_Bonds[Elements[count_i]]:
                sep,idx = cons.pop(0)
                Adj_mat[count_i,idx] = 0
                Adj_mat[idx,count_i] = 0

    # Print warning messages for obviously suspicious bonding motifs.
    if sum( [ problem_dict[i] for i in problem_dict.keys() ] ) > 0:
        return False
        '''
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                    if i == "H": print( "WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(problem_dict[i]))
                    if i == "C": print( "WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "Si": print( "WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "F": print( "WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Cl": print( "WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "Br": print( "WARNING in Table_generator: {} bromine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "I": print( "WARNING in Table_generator: {} iodine(s) have more than one bond.".format(problem_dict[i]))
                    if i == "O": print( "WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(problem_dict[i]))
                    if i == "N": print( "WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(problem_dict[i]))
                    if i == "B": print( "WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(problem_dict[i]))
        print( "")
        '''
    else:
        return Adj_mat


# Determines the Reactive Atoms 
# Inputs:       Elements: N-element List strings for each atom type
#               R_bond_mat: Reactant bond order matrix.
#               P_bond_mat: Product bond order matrix.
# Returns:      bond_change: list of bond changes and where they are. 
#               bond_broken: list of bonds broken
#               bond_formed: list of bonds formed
#               bond_ochangeup: list of bonds with the order increased
#               bond_ochangedown: list of bonds with the order increase.
#               involve: Sorted set of bond changes
def return_reactive(E,Rbond_mat,Pbond_mat):

    # generate adjacency matrix
    bondmat_change = Pbond_mat - Rbond_mat
    
    # determine breaking and forming bonds
    bond_change  = []
    bond_formed = []
    bond_broken = []
    bond_ochangeup = []
    bond_ochangedown = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if bondmat_change[i][j] != 0:
                bond_change += [(i,j)]
                # If there was no bond at the reactant, state that the bond is formed. 
                if Rbond_mat[i][j] == 0:
                    bond_formed += [(i,j)]
                # If there was no bond at the product, state that it is broken. 
                elif Rbond_mat[i][j] == 0:
                    bond_broken += [(i,j)]
                elif Rbond_mat[i][j] > Pbond_mat[i][j]:
                    bond_ochangedown += [(i,j)]
                elif Rbond_mat[i][j] < Pbond_mat[i][j]:
                    bond_ochangeup += [(i,j)]

    involve = sorted(list(set(list(sum(bond_change, ())))))

    return bond_change,involve,bond_formed,bond_broken,bond_ochangeup,bond_ochangedown


# Function that takes a 3D geometry and gets the SMILES 
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 np.array holding the geometry of the molecule
#               Mode:  Optional. The way you want to find smiles. 
# Returns:      SMILES of reactant and product
def getSMILESfrom3D(E,RG,PG,Rind,mode='TAFFI'):

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

# Function that reorganizes the Matrix for the Atom-Mapped smiles. 
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 np.array holding the geometry of the molecule
#               Mode:  Optional. The way you want to find smiles. 
# Returns:      SMILES of reactant and product
def return_matrix(AM_smiles):

    # load in mol
    mol = Chem.MolFromSmiles(AM_smiles,sanitize=False)

    # Get the list of atoms sorted by their atom-mapping number
    sorted_atoms = sorted(mol.GetAtoms(), key=lambda atom: atom.GetAtomMapNum())

    # Create an empty editable molecule
    new_mol = Chem.EditableMol(Chem.Mol())

    # Add the sorted atoms to the new molecule
    atom_map = {}
    for atom in sorted_atoms:
        idx = new_mol.AddAtom(atom)
        atom_map[atom.GetAtomMapNum()] = idx

    # Add the bonds to the new molecule using the atom_map
    for bond in mol.GetBonds():
        begin_atom = atom_map[bond.GetBeginAtom().GetAtomMapNum()]
        end_atom = atom_map[bond.GetEndAtom().GetAtomMapNum()]
        bond_type = bond.GetBondType()
        new_mol.AddBond(begin_atom, end_atom, bond_type)

    # Get the final molecule and remove atom mapping numbers
    new_mol = new_mol.GetMol()
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(0)

    # Get the number of atoms in the molecule
    num_atoms = new_mol.GetNumAtoms()

    # Initialize an empty bond matrix with zeros
    adj_mat = np.zeros((num_atoms, num_atoms), dtype=int)
    bond_mat= np.zeros((num_atoms, num_atoms), dtype=int)
    fc      = []
    element = []
    # Fill in the bond matrix with bond orders (1 for single, 2 for double, etc.)
    for bond in new_mol.GetBonds():
        # obtain index
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1
        
        # obtain bond order
        bond_order = int(bond.GetBondTypeAsDouble())
        bond_mat[i, j] = bond_order
        bond_mat[j, i] = bond_order

    # Iterate over the atoms in the molecule and print formal charges
    for atom in new_mol.GetAtoms():
        formal_charge = atom.GetFormalCharge()
        fc.append(formal_charge)
        element.append(atom.GetSymbol())

    return element, adj_mat, bond_mat, fc

# Function that Finds the Bond Stereochemistry from the Atom-mapped SMILES. 
# Inputs:       Atom-mapped SMILES. 
# Returns:      SMILES of reactant and product
def find_stereochemistry(AM_smiles):

    # initialize 
    bond_stereo    = {}
    chiral_centers = {}
    Hybridization  = {}
    conjugation    = {}
    atom_aromatic,bond_aromatic = {},{}

    # generate two mols, one with origianl sequence and one wit sanitize
    mol = Chem.MolFromSmiles(AM_smiles, sanitize=False)
    mol_sanitized = Chem.MolFromSmiles(AM_smiles)

    # Get the atom chiral centers from the sanitized molecule
    chiral_centers_sanitized = Chem.FindMolChiralCenters(mol_sanitized, includeUnassigned=True)

    # Get the bond stereo information from the sanitized molecule
    bond_stereo_info = {}
    for bond in mol_sanitized.GetBonds():
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom = bond.GetEndAtom().GetAtomMapNum()
        bond_index = tuple(sorted([begin_atom,end_atom]))
        conjugation[bond_index] = bond.GetIsConjugated()
        bond_aromatic[bond_index] = bond.GetIsAromatic()

        stereo = bond.GetStereo()
        if stereo != Chem.BondStereo.STEREONONE:
            bond_stereo_info[(begin_atom, end_atom)] = stereo

    # go through heavy atoms
    for atom in mol_sanitized.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        atom_aromatic[atom_map_num] = atom.GetIsAromatic()
        Hybridization[atom_map_num] = atom.GetHybridization()

    # obtain chiral centers
    for chiral_center in chiral_centers_sanitized:
        atom_index, chirality = chiral_center
        atom_map_num = mol_sanitized.GetAtomWithIdx(atom_index).GetAtomMapNum()
        chiral_centers[atom_map_num] = chirality

    # Compare with the original molecule
    for bond in mol.GetBonds():
        
        begin_atom = bond.GetBeginAtom().GetAtomMapNum()
        end_atom   = bond.GetEndAtom().GetAtomMapNum()
        bond_index = tuple(sorted([begin_atom,end_atom]))

        if (begin_atom, end_atom) in bond_stereo_info:
            stereo = bond_stereo_info[(begin_atom, end_atom)]

            if stereo == Chem.BondStereo.STEREOE:
                bond_stereo[bond_index] = 'E'
            elif stereo == Chem.BondStereo.STEREOZ:
                bond_stereo[bond_index] = 'Z'
            elif stereo == Chem.BondStereo.STEREOANY:
                bond_stereo[bond_index] = 'ANY'

    return chiral_centers, atom_aromatic, Hybridization, bond_stereo, conjugation, bond_aromatic

# Function that Finds the Break N From N Amount. 
# Inputs:       Elements: N-element List strings for each atom type
#               Bond Matrix: Nx3 np.array holding the bond order of the molecule
#               
# Returns:      No. of Bond breaks and formations
def return_bnfn(E,Rbond_mat,Pbond_mat):

    # generate adjacency matrix
    bondmat_change = Pbond_mat - Rbond_mat
    
    # determine breaking and forming bonds
    bond_break  = []
    bond_form  = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if bondmat_change[i][j] < 0:
                bond_break += [(i,j)]
            elif bondmat_change[i][j] > 0:
                bond_form += [(i,j)]

    
    return len(bond_break),len(bond_form)




