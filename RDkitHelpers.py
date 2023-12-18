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

from rdkit.Chem import inchi



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
    mol = Chem.MolFromSmiles(smi)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return mol


# Determines the Inchi key from SMILES   
# Inputs:       smi: SMILES string
# Returns:      Inchi key
def getInchifromSMILES(smi):
    mol = Chem.MolFromSmiles(smi)
    # Generate the InChI
    inchi_str = inchi.MolToInchi(mol)

    # Generate the InChIKey from InChI
    inchi_key = inchi.InchiToInchiKey(inchi_str)
    return inchi_str



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


