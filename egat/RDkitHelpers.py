from ast import excepthandler
import sys
sys.path.append('utilities')
import json
import os,sys,subprocess
import numpy as np
import pandas as pd
import argparse 
import traceback    
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdDistGeom
from rdkit.Chem import inchi

import oddt
from oddt import interactions



# Determines the SMILES from a mol file.  
# Inputs:       molfile: .mol file
# Returns:      SMILES string
def return_smi(molfile):
    # convert mol file into rdkit mol onject
    mol = Chem.rdmolfiles.MolFromMolFile(molfile,removeHs=False)
    return Chem.MolToSmiles(mol)

# Determines the Atom-mapped SMILES from a mol file.  
# Inputs:       molfile: .mol file
# Returns:      SMILES string
def return_atommaped_smi(molfile):

    # convert mol file into rdkit mol onject
    mol = Chem.rdmolfiles.MolFromMolFile(molfile,removeHs=False)

    # assign atom index
    mol = mol_with_atom_index(mol)

    return Chem.MolToSmiles(mol)

# Function to assign atom index to each atom in the mol file
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms): mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()+1))
    return mol


def AddHMapping(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['H','Cl','F','I','Br']:
            atom.SetAtomMapNum(atom.GetIdx() + 1)  # Increment the atom mapping label for H atoms
    return Chem.MolToSmiles(mol)


def RemoveMapping(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    except:
        mol = Chem.MolFromSmiles(smi,sanitize=False)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return mol


# Determines the Inchi key from SMILES   
# Inputs:       smi: SMILES string
# Returns:      Inchi key
def getInchifromSMILES(smi):
    mol = Chem.MolFromSmiles(smi)
    # Generate the InChI
    try:
        inchi_str = inchi.MolToInchi(mol)

        # Generate the InChIKey from InChI
        inchi_key = inchi.InchiToInchiKey(inchi_str)
    except:
        inchi_key = 'InChI_NA'
    return inchi_key



# Function that Finds the Bond Stereochemistry from the Atom-mapped SMILES. 
# Inputs:       Atom-mapped SMILES. 
# Returns:      Heavu Atom Count
def getHA(smi):
    mol = Chem.MolFromSmiles(smi)
    HAcount = mol.GetNumHeavyAtoms()
    return HAcount



# Function that Finds the Location of Rotatable Bonds. 
# Inputs:       Atom-mapped SMILES. 
# Returns:      Location of Rotatable Bonds
def getRotatableBondCount(AM_smiles):
    mol = Chem.MolFromSmiles(AM_smiles)
    RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    return mol.GetSubstructMatches(RotatableBond)


# Function that Finds the Location of Spiro atoms. Takes the C code from RDKit and gets the location of the atoms.  
# Inputs:       Atom-mapped SMILES. 
# Returns:      Location of Spiro atoms
def GetSpiroAtoms(smi, atoms=None):
    mol = Chem.MolFromSmiles(smi)
    if not mol.getRingInfo() or not mol.getRingInfo().isInitialized():
        rdmolops.findSSSR(mol)
    rInfo = mol.getRingInfo()
    lAtoms = []
    if not atoms:
        atoms = lAtoms

    for i in range(len(rInfo.atomRings())):
        ri = rInfo.atomRings()[i]
        for j in range(i + 1, len(rInfo.atomRings())):
            rj = rInfo.atomRings()[j]
            inter = set(ri).intersection(rj)
            if len(inter) == 1:
                if inter[0] not in atoms:
                    atoms.append(inter[0])
    return atoms

# Function that Finds the Location of Bridgehead atoms. Takes the C code from RDKit and gets the location of the atoms.  
# Inputs:       Atom-mapped SMILES. 
# Returns:      Location of Bridgehead atoms
def GetBridgeheadAtoms(mol, atoms=None):
    if not mol.getRingInfo() or not mol.getRingInfo().isInitialized():
        rdmolops.findSSSR(mol)
    rInfo = mol.getRingInfo()
    lAtoms = []
    if not atoms:
        atoms = lAtoms
    for i in range(len(rInfo.bondRings())):
        ri = rInfo.bondRings()[i]
        for j in range(i + 1, len(rInfo.bondRings())):
            rj = rInfo.bondRings()[j]
            inter = set(ri).intersection(rj)
            if len(inter) > 1:
                atomCounts = [0] * mol.getNumAtoms()
                for ii in inter:
                    atomCounts[mol.getBondWithIdx(ii).getBeginAtomIdx()] += 1
                    atomCounts[mol.getBondWithIdx(ii).getEndAtomIdx()] += 1
                for ti in range(len(atomCounts)):
                    if atomCounts[ti] == 1:
                        if ti not in atoms:
                            atoms.append(ti)




# Dictionary mapping elements to their electronegativity values (Pauling scale)
def readenegtable():
    # Read the content of the file
    with open('/depot/bsavoie/data/Mahit-TS-Energy-Project/GitHub/EGAT/electroneg-pauling.txt', 'r') as file:
        lines = file.readlines()

    # Initialize an empty dictionary to store atomic numbers and electronegativity values
    electronegativity_dict = {}

    # Iterate through the lines and extract relevant information
    for line in lines:
    
        parts = [part.strip() for part in line.split('#') if part.strip()]
        if len(parts) == 2:
            if parts[0] == '-':
                electronegativity = 0
            else:
                electronegativity = float(parts[0])
            
            element = parts[1].split(' ')[1]
            electronegativity_dict[element] = electronegativity
    return electronegativity_dict

def get_electronegativity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    electronegativity_dict = readenegtable()
    symbolsindict = list(electronegativity_dict.keys())
    if mol:
        atoms = mol.GetAtoms()
        electronegativity_list = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            if symbol in symbolsindict:
                electronegativity_list.append((symbol, electronegativity_dict[symbol]))
            else:
                electronegativity_list.append((symbol, 0))
        return electronegativity_list
    else:
        return None

def calculate_bond_polarity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    electronegativity_dict = readenegtable()
    symbolsindict = list(electronegativity_dict.keys())
    if mol:
        bonds = mol.GetBonds()
        polarity_info = []
        for bond in bonds:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_symbol = begin_atom.GetSymbol()
            end_symbol = end_atom.GetSymbol()

            if begin_symbol in symbolsindict  and end_symbol in symbolsindict :
                begin_en = electronegativity_dict[begin_symbol]
                end_en = electronegativity_dict[end_symbol]

                # Calculate electronegativity difference
                en_difference = abs(begin_en - end_en)
                polarity_info.append(((begin_symbol, end_symbol), en_difference))
            else:
                polarity_info.append(((begin_symbol, end_symbol), 0))
        return polarity_info
    else:
        return None

def find_hbond_acceptors_donors(smiles):
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    print(smiles)
    if mol:
        acceptor_atoms = []
        acceptor_atoms_EGAT = []
        donor_atoms = []
        donor_atoms_EGAT = []

        Hdonor_index = []
                
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            neighbors = atom.GetNeighbors()
            
            if symbol == 'O' or symbol == 'N' or symbol == 'F':
                acceptor_atoms.append(atom.GetIdx())
                print(symbol)
                acceptor_atoms_EGAT += [1]
            else:
                acceptor_atoms_EGAT += [0]
            
            if symbol == 'H':
                for neighbor in neighbors:
                    neighbor_symbol = neighbor.GetSymbol()
                    
                    if neighbor_symbol == 'O' or neighbor_symbol == 'N' or neighbor_symbol == 'F':
                        donor_atoms.append(neighbor.GetIdx())
                        Hdonor_index.append(atom.GetIdx())
                        print(symbol)
                        donor_atoms_EGAT += [1]
                    else:
                        donor_atoms_EGAT += [0]

        return acceptor_atoms, donor_atoms,Hdonor_index,acceptor_atoms_EGAT, donor_atoms_EGAT
    else:
        return None, None


def find_intramolecular_hbonds(mol,confId=-1,eligibleAtoms=[7,8,9],distTol=2.5):
    '''
    eligibleAtoms is the list of atomic numbers of eligible H-bond donors or acceptors
    distTol is the maximum accepted distance for an H bond
    '''
    res = []
    conf = mol.GetConformer(confId)
    for i in range(mol.GetNumAtoms()):
        atomi = mol.GetAtomWithIdx(i)
        # is it H?
        if atomi.GetAtomicNum()==1:
            if atomi.GetDegree() != 1:
                continue
            nbr = atomi.GetNeighbors()[0]
            if nbr.GetAtomicNum() not in eligibleAtoms:
                continue
            # loop over all other atoms except ones we're bound to and other Hs:
            for j in range(mol.GetNumAtoms()):
                if j==i:
                    continue
                atomj = mol.GetAtomWithIdx(j)
                if atomj.GetAtomicNum() not in eligibleAtoms or mol.GetBondBetweenAtoms(i,j):
                    continue
                dist = (conf.GetAtomPosition(i)- conf.GetAtomPosition(j)).Length()
                if dist<distTol:
                    res.append((i,j,dist))
    return res


def find_intermolecular_hbonds(smi1,smi2):
    acceptors_mol1,donors_mol1,_,_ = find_hbond_acceptors_donors(smi1)
    acceptors_mol2,donors_mol2,_,_ = find_hbond_acceptors_donors(smi2)
    
    
    
    return None


def determineHbondsIntramolecular(smiles,distTol=2.5):
    mol = Chem.MolFromSmiles(smiles)
    N = mol.GetNumAtoms() 
    rdDistGeom.EmbedMolecule(mol)
    rdDistGeom.EmbedMultipleConfs(mol,3*N-6, randomSeed=10) #-> Map conformers by degrees of Freedom.
    acceptors,donors,_,_ = find_hbond_acceptors_donors(smiles)
    
    Hbondsset = [find_intramolecular_hbonds(mol,_,distTol) for _ in mol.GetNumConformers()]
    return Hbondsset