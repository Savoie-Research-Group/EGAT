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
from xtb_functions import xtb_energy,xtb_geo_opt
from taffi_functions import *
from job_submit import *
from utility import *
from xgboost import XGBClassifier
import rdkit 
from rdkit import Chem
import pandas as pd


def main_fcn(csvfile='preppedrxns_xTB.csv',folder = '/depot/bsavoie/data/YARP_database/model_reaction/reaction_db'):
    df = pd.read_csv(folder+'/cleaned_csv_files/' + csvfile)
    mappedRxns = []
    for i in range(len(df)):
        if df['type'][i] == 'intended':
            fname = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + '.xyz'
            input_xyz = folder + '/' + df['folder'][i] + '/input_files_conf/' + fname 
            E,RG,PG,Radj_mat,Padj_mat = parse_input(input_xyz,return_adj=True)
        else:
            fname = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + '-IRC.xyz'
            input_xyz = folder + '/' + df['folder'][i] + '/low-IRC-result/' + fname 
            print('UNINTENDED')
            E1,RG1,PG1 = parse_input(input_xyz,return_adj=False)
            E,RG,PG,Radj_mat,Padj_mat = parse_IRC_xyz(input_xyz,len(E1))
    
        namespace = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + '_reactant'
        mol_write("{}_input.mol".format(namespace), E, RG, Radj_mat)
        # convert mol file into rdkit mol onject
        mol=Chem.rdmolfiles.MolFromMolFile("{}_input.mol".format(namespace),removeHs=False)
        # assign atom index
        try:
            mol=mol_with_atom_index(mol)
            mappedReactants = Chem.MolToSmiles(mol)
        except AttributeError:
            print('MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + " Cannot Map with TAFFI-based technique")
            mol_file = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + '_input.mol'
            os.system('obabel {} -O {} -h --gen3d best --minimize --sd --ff UFF'.format(fname,mol_file))
            mol = Chem.rdmolfiles.MolFromMolFile(mol_file,removeHs=False)
            try: 
                mol = mol_with_atom_index(mol)
                mappedReactants = Chem.MolToSmiles(mol)
            except AttributeError:
                mappedReactants = "Cannot Map"
        os.system("rm {}".format(namespace + '_input.mol'))

        namespace = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + '_product'
        mol_write("{}_input.mol".format(namespace), E, PG, Padj_mat)
        # convert mol file into rdkit mol onject
        mol=Chem.rdmolfiles.MolFromMolFile("{}_input.mol".format(namespace),removeHs=False)
        # assign atom index
        try:
            mol=mol_with_atom_index(mol)
            mappedReactants = Chem.MolToSmiles(mol)
        except AttributeError:
            print('MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + " Cannot Map with TAFFI-based technique")
       	    mol_file = 'MR_' + str(df['channel'][i]) + '_' + str(df['subtype'][i]) + '_input.mol'
            os.system('obabel {} -O {} -h --gen3d best --minimize --sd --ff UFF'.format(fname,mol_file))
            mol	= Chem.rdmolfiles.MolFromMolFile(mol_file,removeHs=False)
       	    try: 
       	       	mol = mol_with_atom_index(mol)
       	       	mappedReactants	= Chem.MolToSmiles(mol)
       	    except AttributeError:
       	       	mappedReactants	= "Cannot Map"        
        os.system("rm {}".format(namespace + '_input.mol'))
        print(mappedReactants + '>>' + mappedProducts)
        mappedRxns += [mappedReactants + '>>' + mappedProducts]
    
    df2 = pd.DataFrame() 
    df2['mappedrxns'] = mappedRxns
    df2['Ea'] = df['Ea']
    df2.to_csv(folder+'/cleaned_csv_files/' + 'preppedrxns_xTB2.csv')

def parse_IRC_xyz(IRC_xyz,Natoms):
    with open(IRC_xyz,'r') as g: lines = g.readlines()

    count = 0
    write_reactant= []
    write_product = []
    Energy_list   = []
    N_image = int(len(lines)/(Natoms+2))

    for lc,line in enumerate(lines):
        fields = line.split()
        if len(fields)==1 and fields[0] == str(Natoms): count += 1
        if len(fields) == 3 and 'energy' in fields[2]: Energy_list += [float(fields[2].split(':')[-1])]
        if 'Coordinates' in fields and 'E' in fields: Energy_list += [float(fields[-1])]
        if count == 1: write_reactant+= [line]
        if count == N_image: write_product += [line]

        Energy_list = np.array(Energy_list)

    # write the reactant and product
    with open(IRC_xyz.replace('-IRC.xyz','-start.xyz'),'w') as g:
        for line in write_reactant: g.write(line)

    # parse IRC start point xyz file
    NE,NG1  = xyz_parse(IRC_xyz.replace('-IRC.xyz','-start.xyz'))
    N_adj_1 = Table_generator(NE,NG1)

    # generate end point of IRC
    with open(IRC_xyz.replace('-IRC.xyz','-product.xyz'),'w') as g:
        for line in write_product: g.write(line)

    # parse IRC start point xyz file
    NE,NG2  = xyz_parse(IRC_xyz.replace('-IRC.xyz','-product.xyz'))
    N_adj_2 = Table_generator(NE,NG2)

    os.system("rm {} {}".format(IRC_xyz.replace('-IRC.xyz','-start.xyz'),IRC_xyz.replace('-IRC.xyz','-product.xyz')))

    return NE,NG1,NG2,N_adj_1,N_adj_2

# Function to parse pyGSM input files
def parse_input(input_xyz,return_adj=False):

    name = input_xyz.split('/')[-1].split('xyz')[0]
    xyz  = ['','']
    count= 0

    # read in pairs of xyz file
    with open(input_xyz,"r") as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0:
                N = int(fields[0])
                xyz[0] += lines
                continue

            if len(fields) == 1 and float(fields[0]) == float(N):
                count+=1
            if count > 1:
                xyz += ['']
            xyz[count]+=lines

    with open('{}_reactant.xyz'.format(name),"w") as f:
        f.write(xyz[0])

    with open('{}_product.xyz'.format(name),"w") as f:
        f.write(xyz[1])

    # parse reactant info
    E,RG   = xyz_parse('{}_reactant.xyz'.format(name))

    # parse product info
    _,PG   = xyz_parse('{}_product.xyz'.format(name))

    try:
        os.remove('{}_reactant.xyz'.format(name))
        os.remove('{}_product.xyz'.format(name))
    except:
        pass

    if return_adj:
        # generate adj_mat if is needed
        Radj_mat = Table_generator(E, RG)
        Padj_mat = Table_generator(E, PG)
        return E,RG,PG,Radj_mat,Padj_mat

    else:
        return E,RG,PG

def mol_write(name,elements,geo,adj_mat,q=0,append_opt=False):

    # Consistency check
    if len(elements) >= 1000:
        print( "ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return
    
    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
    	open_cond = 'w'
    
    # Parse the basename for the mol header
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]
    
    lones,bondings,cores,bond_mat,fc = find_lewis(elements,adj_mat,q_tot=q,keep_lone=[],return_pref=False,return_FC=True)
    bond_mat = bond_mat[0]
    
    # deal with radicals
    keep_lone = [ [ count_j for count_j,j in enumerate(lone_electron) if j%2 != 0] for lone_electron in lones][0]
    
    # deal with charges
    fc = fc[0]
    chrg = len([i for i in fc if i != 0])
    
    # Write the file
    with open(name,open_cond) as f:
        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))
        
        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(np.sum(adj_mat/2.0))))
        
        # Write the geometry
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2],i))
        
        # Write the bonds
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ]
        for i in bonds:
        
            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0],i[1]])
        
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))
        
        # write radical info if exist
        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(1,keep_lone[0]+1,2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(2,keep_lone[0]+1,2,keep_lone[1]+1,2))
            else:
                print("Only support one/two radical containing compounds, radical info will be skip in the output mol file...")
        
        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(1,fc.index(charge)+1,charge))
            else:
                info = "M  CHG{:>3d}".format(chrg)
                for count_c,charge in enumerate(fc):
                    if charge != 0: info += '{:>4d}{:>4d}'.format(count_c+1,charge)
                info += '\n'
                f.write(info)
    
        f.write("M  END\n$$$$\n")
    
    return

if __name__ == "__main__":
    main_fcn()

