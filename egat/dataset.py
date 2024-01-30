'''
Author: Qiyuan Zhao, Sai Mahit Vaddadi

Last Edit: November 30,2023
'''
from audioop import add
from selectors import EpollSelector
import sqlite3
from this import d
import numpy as np
import os
from torch.utils.data import Dataset
import dgl
import torch
import json
from tqdm import tqdm
import pandas as pd 
import random
import pymongo
import sqlite3
import omegaconf
def TurnSQLdfintoDict(rawdata,target,additional):
        shouldbeints = ['Indices']
        shouldbefloats = []
        if isinstance(target,list): shouldbefloats += target
        else: shouldbefloats += [target]

        if isinstance(additional,list): shouldbefloats += additional
        elif isinstance(additional,str): shouldbefloats += [additional]

        shouldbearrays = ['u','v','atom_F_R','atom_F_P','bond_F_R','bond_F_P','R_Addon','P_Addon']

        outdict = dict()
        for row in rawdata.index.tolist():
            key = row
            value = rawdata.loc[key,'value']
            
            if key in shouldbeints:
                outdict[key] = int(value)
            elif key in shouldbefloats:
                outdict[key] = float(value)
            elif key in shouldbearrays:
                outdict[key] = np.array(eval(value))
            else:
                outdict[key] = value
        return outdict

class RGD1Dataset(Dataset):
    """
    Class that takes the .json folder into a Torch dataset for use by the model. 
    Parameters
    ----------
    root: string
            Folder to look at. 
    npoints: int
            Maximum point in each molecule in the database (in RGD1 max_point = 33)
    split: string
            Split to look at. 
    class_choice: string or list
            Types of reactions to look at.
    exclude: list
            Reactions to not look at. 
    randomize:
            Shuffle the data so that the batches are loaded with different reaction classes along with shuffling the batches themselves
    fold: int
            Fold to look at
    foldtype: string
            Fold type to look at
    size: int
            Look at the first N samples
    target: string or list
            Target column or set of columns to predict
    additional: string or list
            Column or set of columns to use for predictions. 
    hasaddons: bool
            Checks if RDKit global features need to be used
    molecular: bool
            Checks if the molecular setting is used. 

    Returns
    -------
    DataLoader: torch.DataLoader
    """

    

    def __init__(self, root='./data/RGD1', npoints=33, split='train', class_choice=None,exclude=[],randomize=False,fold=None,foldtype=None,size = None,target='DE',additional='DH',hasaddons=False,molecular=False,datastorage=None):
        
        ###### LOAD VARIABLES
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'Rxntype.txt')
        self.cat = []
        self.target = target
        self.additional = additional
        self.addons = hasaddons
        self.molecular =molecular
        

        ###### LOAD CATFILE
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat.append(ls[0])

        ##### CHECK IF THE DATASET IS A BUNCH OF JSON OR A DATABASE
        if datastorage is None:
            ###### LOAD THE REACTION CLASSES WE WANT TO LOOK AT
            if not class_choice is None:
                self.cat = [k for k in self.cat if k in class_choice]

            ###### LOAD THE TRAIN-VAL-TEST SPLIT
            if fold is None:
                with open(os.path.join(self.root, 'train_test_split', 'train_idx.json'), 'r') as f:
                    train_ids = set([str(d.split('/')[-1]) for d in json.load(f) if str(d.split('/')[-1]) not in exclude])
                with open(os.path.join(self.root, 'train_test_split', 'val_idx.json'), 'r') as f:
                    val_ids = set([str(d.split('/')[-1]) for d in json.load(f) if str(d.split('/')[-1]) not in exclude])
            
                with open(os.path.join(self.root, 'train_test_split','test_idx.json'), 'r') as f:
                    test_ids = set([str(d.split('/')[-1]) for d in json.load(f) if str(d.split('/')[-1]) not in exclude])
            
            else:
                with open(os.path.join(self.root, 'train_test_split', foldtype,f'fold{fold}','train_idx.json'), 'r') as f:
                    train_ids = set([str(d.split('/')[-1]) for d in json.load(f) if str(d.split('/')[-1]) not in exclude])
                with open(os.path.join(self.root, 'train_test_split',foldtype,f'fold{fold}','val_idx.json'), 'r') as f:
                    val_ids = set([str(d.split('/')[-1]) for d in json.load(f) if str(d.split('/')[-1]) not in exclude])
                
                with open(os.path.join(self.root, 'train_test_split',foldtype,f'fold{fold}','test_idx.json'), 'r') as f:
                    test_ids = set([str(d.split('/')[-1]) for d in json.load(f) if str(d.split('/')[-1]) not in exclude])

            ###### LOAD THE TRAIN-VAL-TEST DATA
            self.meta = {}
            for item in self.cat:
                self.meta[item] = []
                dir_point = os.path.join(self.root, item)
                fns = sorted(os.listdir(dir_point))

                ###### LOAD THE DATA FOR THE RELEVANT SPLIT
                if split == 'trainval':
                    allsets = train_ids.union(train_ids,val_ids)
                    fns = [fn for fn in fns if fn[0:-5] in allsets]
                elif split == 'train':
                    fns = [fn for fn in fns if fn[0:-5] in train_ids]
                elif split == 'val':
                    fns = [fn for fn in fns if fn[0:-5] in val_ids]
                elif split == 'test':
                    fns = [fn for fn in fns if fn[0:-5] in test_ids]
                elif split == 'all':
                    allsets = train_ids.union(val_ids,test_ids)
                    fns = [fn for fn in fns if fn[0:-5] in allsets]
                elif split == 'valtest':
                    allsets = val_ids.union(test_ids)
                    fns = [fn for fn in fns if ((fn[0:-4] in val_ids) or (fn[0:-4] in test_ids))]
                    fns = [fn for fn in fns if fn[0:-5] in allsets]
                elif split == 'traintest':
                    allsets = train_ids.union(train_ids,test_ids)
                    fns = [fn for fn in fns if fn[0:-5] in allsets]
                else:
                    print('Unknown split: %s. Exiting..' % (split))
                    exit(-1)

                for fn in fns:
                    token = (os.path.splitext(os.path.basename(fn))[0])
                    self.meta[item].append(os.path.join(dir_point, token + '.json'))
              ###### APPEND DATA FROM EACH CLASS INTO THE DATAPATH
              # first element in datapath is class
            self.datapath = []
            for item in self.cat:
                for fn in self.meta[item]:
                    self.datapath.append((item, fn))
        elif datastorage == 'SQL':
            # Get the tables with the relevant class_choice
            self.datapath = []
            conn = sqlite3.connect(os.path.join(root,root.split('/')[-1]+'.db'))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            self.rtypes = []  
           
            for table in tables: 
                if table[0] != 'sqlite_sequence':
                    df = pd.read_sql_query(f"SELECT * FROM {table[0]}", conn)
                    df = df.set_index('key')
                    df = df[~df.index.duplicated(keep='first')]
                    if isinstance(class_choice,list):
                        if df.loc['rxntype','value'] in class_choice:
                            if split in ['train','test','val']:
                                if fold is None:
                                    if df.loc['split','value'] == split:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] == split:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'trainval':
                                if fold is None:
                                    if df.loc['split','value'] in ['train','val']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['train','val']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'traintest':
                                if fold is None:
                                    if df.loc['split','value'] in ['train','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['train','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'testval':
                                if fold is None:
                                    if df.loc['split','value'] in ['val','test']:
                                        self.datapath.append(table[0])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['val','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'all':
                                if fold is None:
                                    if df.loc['split','value'] in ['train','val','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['train','val','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                    elif isinstance(class_choice,str):
                        if df.loc['rxntype','value'] == class_choice:
                            if split in ['train','test','val']:
                                if fold is None:
                                    if df.loc['split','value'] == split:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] == split:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'trainval':
                                if fold is None:
                                    if df.loc['split','value'] in ['train','val']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['train','val']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'traintest':
                                if fold is None:
                                    if df.loc['split','value'] in ['train','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['train','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'testval':
                                if fold is None:
                                    if df.loc['split','value'] in ['val','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['val','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                            elif split == 'all':
                                if fold is None:
                                    if df.loc['split','value'] in ['train','val','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                                else:
                                    if df[f'fold{fold}_split','value'] in ['train','val','test']:
                                        self.datapath.append(table[0])
                                        self.rtypes.append(df.loc['rxntype','value'])
                    else:
                        if split in ['train','test','val']:
                            if fold is None:
                                #print(df.loc['split','value'])
                                if df.loc['split','value'] == split:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                            else:
                                if df[f'fold{fold}_split','value'] == split:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                        elif split == 'trainval':
                            if fold is None:
                                if df.loc['split','value'] in ['train','val']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                            else:
                                if df[f'fold{fold}_split','value'] in ['train','val']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                        elif split == 'traintest':
                            if fold is None:
                                if df.loc['split','value'] in ['train','test']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                            else:
                                if df[f'fold{fold}_split','value'] in ['train','test']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                        elif split == 'testval':
                            if fold is None:
                                if df.loc['split','value'] in ['val','test']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                            else:
                                if df[f'fold{fold}_split','value'] in ['val','test']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                        elif split == 'all':
                            if fold is None:
                                if df.loc['split','value'] in ['train','val','test']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])
                            else:
                                if df[f'fold{fold}_split','value'] in ['train','val','test']:
                                    self.datapath.append(table[0])
                                    self.rtypes.append(df.loc['rxntype','value'])


            conn.close()
        
        
        ###### GET THE FIRST N SAMPLES
        if datastorage is None:
            ###### SHUFFLE THE LIST IF WE WANT TO RANDOMIZE THE DATA OR GET THE FIRST N SAMPLES
            state = 42
            random.seed(state)
            if randomize: random.shuffle(self.datapath)
            if size is not None: random.shuffle(self.datapath)
            if size is not None:
                df = pd.DataFrame(self.datapath,columns = ['R','N'])
                df = df.iloc[:size,:]
                self.datapath = df['N'].tolist()  
                self.rtypes = df['R'].tolist()      
                self.cache = {}  # from index to (point_set, cls, seg) tuple
                self.cache_size = 1000
            else:
                self.datapath = df['N'].tolist()  
                self.rtypes = df['R'].tolist()      
                self.cache = {}  # from index to (point_set, cls, seg) tuple
                self.cache_size = 1000
                
        elif datastorage == 'SQL':
            if size is not None:
                tmp = pd.DataFrame({'N':self.datapath,'R':self.rtypes})
                tmp = tmp.sample(frac=1)
                tmp = tmp.iloc[:size,:]
                self.datapath = tmp['N'].tolist()  
                self.rtypes = tmp['R'].tolist() 
                self.cache = {}  # from index to (point_set, cls, seg) tuple
                self.cache_size = 1000
            else:
                if randomize: 
                    tmp = pd.DataFrame({'N':self.datapath,'R':self.rtypes})
                    tmp = tmp.sample(frac=1)
                    self.datapath = tmp['N'].tolist()  
                    self.rtypes = tmp['R'].tolist()
                    self.cache = {}  # from index to (point_set, cls, seg) tuple
                    self.cache_size = 1000
                else:
                    self.cache = {}  # from index to (point_set, cls, seg) tuple
                    self.cache_size = 1000
                    
            

        
        self.datastorage = datastorage
        self.root = root

    def __getitem__(self, index):
        
        if index in self.cache:
            samples = self.cache[index]
        else:
            ###### LOAD DATAPATH 
            fn = self.datapath[index]
            rtype = self.rtypes[index]
            if self.datastorage is None:
                if type(fn) is str:
                    with open(fn,'r') as f: info = json.load(f)
                else:
                    with open(fn[1],'r') as f: info = json.load(f)
            elif self.datastorage == 'SQL':
                conn = sqlite3.connect(os.path.join(self.root,self.root.split('/')[-1]+'.db'))
                cursor = conn.cursor()
                df = pd.read_sql_query(f"SELECT * FROM {fn}", conn)
                df = df.set_index('key')

                info = TurnSQLdfintoDict(df,self.target,self.additional)
                
                conn.close()
            
            ###### CREATE GRAPH
            u, v = torch.Tensor(info['u']).int(),torch.Tensor(info['v']).int()
            gR   = dgl.graph((u,v))
            gP   = dgl.graph((u,v))

            ###### APPEND FEATURES
            if not self.molecular :
                gR.ndata['x'] = torch.Tensor(info['atom_F_R'])
                gR.edata['x'] = torch.Tensor(info['bond_F_R'])
                gP.ndata['x'] = torch.Tensor(info['atom_F_P'])
                gP.edata['x'] = torch.Tensor(info['bond_F_P'])

                if gR.ndata['x'].max() > 50 or gP.ndata['x'].max() > 50: 
                    with open('check.txt','a') as ff:
                        ff.write('{}\n'.format(fn[1]))
            else:
                
                #print(torch.Tensor(info['atom_F_R']).shape)
                
                try:
                    gR.ndata['x'] = torch.Tensor(info['atom_F_R'])
                    gR.edata['x'] = torch.Tensor(info['bond_F_R'])
                except:
                    print('smiles: ',info['Rsmiles'])
                    print('No. of Nodes Inputted:', u)
                    gR.ndata['x'] = torch.Tensor(info['atom_F_R'])
                    gR.edata['x'] = torch.Tensor(info['bond_F_R'])
                    
                if gR.ndata['x'].max() > 50: 
                    with open('check.txt','a') as ff:
                        ff.write('{}\n'.format(fn[1]))

            ###### MAKE THE OUTPUT TENSORS
            # Add name of the file and the reaction class it inherits
            samples  = []
            if type(fn) is str:
                samples += [fn,rtype]
            else:
                samples += [fn[1],fn[0]]

            # Add the graph
            if not self.molecular:
                samples += [gR,gP]
            else:
                samples += [gR]
            #print(list(info.keys()))
            # Add what the SMILES and Inchi keys are
            if not self.molecular :
                samples += [[info['Rsmiles'],info['Psmiles'],info['Rinchi'],info['Pinchi']]]
            else:
                samples += [[info['Rsmiles'],info['Rinchi']]]

            # Add what the target tensor is
            #targettensor = [gR.number_of_nodes(),gR.number_of_edges()]
            targettensor = []
            
            if isinstance(self.target,list) or isinstance(self.target,omegaconf.listconfig.ListConfig):
                targettensor += [float(info[output]) for output in self.target]
            else:
                targettensor += [float(info[self.target])]
            targettensor = torch.Tensor(targettensor)
            samples += [targettensor]
            
            # Add what the Additional Tensor is if needed
            if self.additional is not None:
                additionaltensor = []
                if isinstance(self.additionals,list) or isinstance(self.additionals,omegaconf.listconfig.ListConfig):
                    additionaltensor += [float(info[output]) for output in self.additionals]
                else:
                    additionaltensor += [float(info[self.additionals])]
                additionaltensor = torch.Tensor(additionaltensor)
                samples += [additionaltensor]

            # Add what the Add-Ons are
            if self.addons:
                if not self.molecular :
                    samples += [torch.Tensor(info['R_Addon']),torch.Tensor(info['P_Addon'])]
                else:
                    samples += [torch.Tensor(info['R_Addon'])]
                
            # save in cache
            if len(self.cache) < self.cache_size:
                self.cache[index] = samples

        return samples

    def __len__(self):
        return len(self.datapath)

# Function to combine graphs
def collatetargetsonly(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    """

    names,types,Rgraphs, Pgraphs, smiles,targets = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    Pbatched_graph = dgl.batch(Pgraphs)
    return names,types,Rbatched_graph, Pbatched_graph, smiles,targets

def collatewitthadditonals(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    additionals: float
            Values being passed in the FFNN
    """
    names,types,Rgraphs, Pgraphs, smiles,targets,additionals = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    Pbatched_graph = dgl.batch(Pgraphs)
    return names,types,Rbatched_graph, Pbatched_graph, smiles,targets,additionals

def collatewithaddons(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    Padd: list
            List of the RDKit Global Features passed from the Product
    """    
    # The input `samples` is a list of pairs (graph, target)
    names,types,Rgraphs, Pgraphs, smiles,targets,Radd,Padd = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    Pbatched_graph = dgl.batch(Pgraphs)
    return names,types,Rbatched_graph, Pbatched_graph, smiles,targets,Radd,Padd

def collateall(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    Pbatched_graph: DGL graph
            Graph of the Product
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    Padd: list
            List of the RDKit Global Features passed from the Product
    additionals: float
            Values being passed in the FFNN
    """    
    # The input `samples` is a list of pairs (graph, target)
    names,types,Rgraphs, Pgraphs, smiles,targets,additionals,Radd,Padd = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    Pbatched_graph = dgl.batch(Pgraphs)
    return names,types,Rbatched_graph, Pbatched_graph,smiles,targets,additionals,Radd,Padd

def molecularcollatetargetsonly(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    """    
    # The input `samples` is a list of pairs (graph, target)
    names,types,Rgraphs, smiles,targets = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    return names,types,Rbatched_graph, smiles,targets

def molecularcollatewitthadditonals(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    additionals: float
            Values being passed in the FFNN
    """
    # The input `samples` is a list of pairs (graph, target)
    names,types,Rgraphs, smiles,targets,additionals = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    return names,types,Rbatched_graph, smiles,targets,additionals

def molecularcollatewithaddons(samples): 
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    """   
    # The input `samples` is a list of pairs (graph, target)
    names,types,Rgraphs, smiles,targets,Radd,Padd = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    return names,types,Rbatched_graph, smiles,targets,Radd,Padd

def molecularcollateall(samples):
    """
    Function that takes the dataset and gets the batched data. 
    Parameters
    ----------
    samples: torch.DataLoader

    Returns
    -------
    names: string
            Names of the .json files being looked at
    types: string
            Names of the reaction type of the .json file
    Rbatched_graph: DGL graph
            Graph of the reactant
    smiles: string
            Smiles of the Reactant and Product
    targets: float
            Target values
    Radd: list
            List of the RDKit Global Features passed from the Reactant
    additionals: float
            Values being passed in the FFNN
    """    
    # The input `samples` is a list of pairs (graph, target)
    names,types,Rgraphs, smiles,targets,additionals,Radd,Padd = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    return names,types,Rbatched_graph, smiles,targets,additionals,Radd,Padd


if __name__ == '__main__':
    
    data = RGD1Dataset(root='/depot/bsavoie/data/Qiyuan/TS_pred/GAT/RGD1-VB',split='train',class_choice=None,size =10000)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False, num_workers=10, collate_fn=collateall)
    names2 = []
    for name,Rgs,Pgs,targets in tqdm(DataLoader, total=len(DataLoader), smoothing=0.9):
        #count = 1
        names2 += name
    names2 = pd.Series(names2)
    names2 = names2.str.split('.',expand=True).iloc[:,0]
    names2 = names2.str.split('/',expand=True).iloc[:,-1]
    names2.to_csv('sample.csv')
    