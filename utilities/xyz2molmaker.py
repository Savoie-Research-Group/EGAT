#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:20:36 2022

@author: svaddadi
"""
import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import pickle,json
import numpy as np

# Load modules in same folder
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd





def main(argv):
    parser = argparse.ArgumentParser(description='Use to generate a set of xyz2mol jobs')
    parser.add_argument('-csv',dest='name',default='preppedrxns_xTB.csv',help = 'CSV to look at')
    parser.add_argument('-o',dest='output',default='preppedrxns_xTB_xyz2mol.csv',help = 'CSV to save as')
    parser.add_argument('-folder',dest='folder',default='/depot/bsavoie/data/YARP_database/model_reaction/reaction_db',
                        help = 'Folder to look at')
    parser.add_argument('-temp',dest='yarp',default='~/YARP/version2.0/utilities',help = 'YARP utilities to look at')
    parser.add_argument('-xyz2mol',dest='xyz2molpath',default='~/YARP/version2.0/utilities/xyz2mol/xyz2mol.py',help = 'location of xyz2mol.py')
    parser.add_argument('-obabel',dest='obabelpath',default='/depot/bsavoie/apps/openbabel_3.1.1/bin/obabel',help = 'OpenBabel directory')
    parser.add_argument('-xyz',dest='xyz',default='/depot/bsavoie/data/YARP_database/model_reaction/reaction_db',
                        help = 'Folder to look at')    
    args=parser.parse_args()
    main_fcn(args.name,args.folder,args.yarp,args.xyz2molpath,args.obabelpath,args.output,args.xyz)
    
    
    
    


def read_xyz_file(xyz_file,namespace):
    file = open(xyz_file,'r')
    lines = file.readlines()
    atomsize = int(lines[0])
    reactant = lines[:atomsize + 2]
    product = lines[-atomsize -2:]
     
    rfile = open("{}_reactant.xyz".format(namespace),'w')
    print('writing {}/{}'.format(os.getcwd(),namespace))
    for i in range(len(reactant)):
        if i != 1:
            rfile.write(reactant[i])
        else:
            rfile.write('\n')
    rfile.close()
    pfile = open("{}_product.xyz".format(namespace),'w')
    print('writing {}/{}'.format(os.getcwd(),namespace))
    for i in range(len(product)):
        if i != 1:
            pfile.write(product[i])
        else:
            pfile.write('\n')
    pfile.close()
    
    
def main_fcn(csvfile='preppedrxns_xTB.csv',folder = '/depot/bsavoie/data/YARP_database/model_reaction/reaction_db',
             yarp = '~/YARP/version2.0/utilities',xyz2molpath = '~/YARP/version2.0/utilities/xyz2mol',
             obabelpath='/depot/bsavoie/apps/openbabel_3.1.1/bin/obabel',output='preppedrxns_xTB_xyz2mol.csv',xyzfolder = '/depot/bsavoie/data/YARP_database/model_reaction/reaction_db'):
    print(csvfile)
    df = pd.read_csv(os.path.join(folder,csvfile))
    mappedRxns = []
    mol2xyzrxns = []
    zwit_React,zwit_Prod = [],[]
    rad_React, rad_Prod = [],[]
    birad_React, birad_Prod = [],[]
    print(yarp)
    os.chdir(yarp)
    print('File will be saved at: {}/{}'.format(yarp,output))
    for i in range(len(df)):
        fname = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + '-IRC.xyz'
        input_xyz = xyzfolder + df['folder'][i] + '/low-IRC-result/' + fname
        print('xyz file name: ' ,input_xyz)
        print()
        folderxyz = folder + '/' + df['folder'][i] + '/low-IRC-result/'
        namespace = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i])
        print('namespace: ' ,namespace)
       	print()
        read_xyz_file(input_xyz,namespace)
                
        os.system('python {} {}.xyz -o sdf > {}.mol'.format(xyz2molpath,namespace+'_reactant',namespace+'_reactant'))
        os.system('python {} {}.xyz -o sdf > {}.mol'.format(xyz2molpath,namespace+'_product',namespace+'_product'))
        print()
        print('smi files in directory')
        os.system('ls *.smi')
        #os.system('cp {}.smi {}_temp.smi'.format(namespace+'_reactant',namespace+'_reactant'))
        #os.system('cp {}.smi {}_temp.smi'.format(namespace+'_product',namespace+'_product'))
        #os.system('{} {}.smi -O {}.mol --gen3d best --minimize --sd --ff UFF'.format(obabelpath,namespace+'_reactant',namespace+'_reactant'))
        #os.system('{} {}.smi -O {}.mol --gen3d best --minimize --sd --ff UFF'.format(obabelpath,namespace+'_product',namespace+'_product'))
        print()
       	print('smi files in directory')
        os.system('ls *.smi')
        print()                 
        #react = open("{}_temp.smi".format(namespace+'_reactant'),'r')
        #rl = react.readlines()
        #print(rl)
        #if len(rl) > 0:
        #    reactsmiles = rl[0]
        #else:
        #    reactsmiles = 'Cannot Map'
        #product = open("{}_temp.smi".format(namespace+'_product'),'r')
        #pl = product.readlines()
        #print(pl)
        #if len(pl) > 0:
        #    productsmiles = pl[0]
       	#else:
       	#    productsmiles = 'Cannot Map'

        mol=Chem.rdmolfiles.MolFromMolFile("{}_reactant.mol".format(namespace),removeHs=False)
        print(type(mol))
        if mol == None:
            os.system('{} -ixyz {}.xyz -osmiles > {}_out.smi'.format(obabelpath,namespace+'_reactant',namespace+'_reactant'))
            file = open('{}_out.smi'.format(namespace+'_reactant'),'r')
            lines = file.readlines()
            if '\t' in lines[0]:
                print(lines[0].split('\t')[0])
                smiles = lines[0].split('\t')[0]
                mol=Chem.rdmolfiles.MolFromSmiles(smiles)
            else:
                print(lines[0].split()[0])
                smiles = lines[0].split()[0]
                mol=Chem.rdmolfiles.MolFromSmiles(smiles)
            if mol == None:
                mappedReactants = 'Cannot Map'
            else:
                mol = mol_with_atom_index(mol)
                mappedReactants = Chem.MolToSmiles(mol)
        else:
            mol = mol_with_atom_index(mol)
            mappedReactants = Chem.MolToSmiles(mol)
        zwit_React += [IsZwit(mappedReactants)]
        rad_React += [IsRadical(mol)]
        birad_React += [IsBiRadical(mol)]

        mol=Chem.rdmolfiles.MolFromMolFile("{}_product.mol".format(namespace),removeHs=False)
        print(type(mol))
        if mol == None:
            os.system('{} -ixyz {}.xyz -osmiles > {}_out.smi'.format(obabelpath,namespace+'_product',namespace+'_product'))
            file = open('{}_out.smi'.format(namespace+'_product'),'r')
            lines = file.readlines()
            print(lines)
            print(lines[0].split())
            if '\t' in lines[0]:
                print(lines[0].split('\t')[0])
       	        smiles = lines[0].split('\t')[0]
       	        mol=Chem.rdmolfiles.MolFromSmiles(smiles)
            else:
                print(lines[0].split()[0])
                smiles = lines[0].split()[0]
                mol=Chem.rdmolfiles.MolFromSmiles(smiles)
            if mol != None:
                mol = mol_with_atom_index(mol)
                mappedProducts = Chem.MolToSmiles(mol)
            else:
                mappedProducts = 'Cannot Map'
        else:   
            mol = mol_with_atom_index(mol)
            mappedProducts = Chem.MolToSmiles(mol)
        zwit_Prod += [IsZwit(mappedProducts)]
        rad_Prod += [IsRadical(mol)]
        birad_Prod += [IsBiRadical(mol)]
        mappedRxns += [mappedReactants + '>>' + mappedProducts]
        #mol2xyzrxns = [reactsmiles + ">>" + productsmiles]
        os.system('rm *.smi')
        os.system('rm *.mol')
        os.system('rm *.xyz')    
    df2 = pd.DataFrame()
    df2['mappedrxns'] = mappedRxns
    #df2['xyz2molrxns'] = mol2xyzrxns
    df2['Ea'] = df['barrier']
    df2['zwit_Prod'] = zwit_Prod
    df2['zwit_React'] = zwit_React
    df2['rad_Prod'] = rad_Prod
    df2['rad_React'] = rad_React    
    df2['birad_Prod'] = birad_Prod
    df2['birad_React'] = birad_React
    if folder[-1] == '/':
        df2.to_csv(folder  + output)
    else:
        df2.to_csv(folder + '/' + output)
# function to assign atom index
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms): mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

def IsZwit(smiles):
    smiles = smiles.split('.')
    checklist = []
    for smile in smiles:
        if '+' in smile:
            if '-' in smile:
                checklist += [True]
            else:
                checklist += [False]
        else:
            checklist += [False]
    if True in checklist:
        return True
    else:
        return False

def IsRadical(mol):
    if mol != None:
        rads = Chem.Descriptors.NumRadicalElectrons(mol)
        if rads > 0:
            return True
        else:
            return False
    else:
        return False

def IsBiRadical(mol):
    if mol != None:
        rads = Chem.Descriptors.NumRadicalElectrons(mol)
        if rads == 2:
            return True
        else:
            return False
    else:
        return False





if __name__ == "__main__":
   main(sys.argv[1:])
