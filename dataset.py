'''
Author: Qiyuan Zhao, Sai Mahit Vaddadi

Last Edit: November 30,2023
'''
from selectors import EpollSelector
import sqlite3
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
import sqllite3

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
        self.catfile = os.path.join(self.root, 'Rxntypes.txt')
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
            conn = sqlite3.connect(root.split('/')[-1]+'.db')
            cursor = conn.cursor()
                
            if not class_choice is None:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table[0]});")
                    columns = cursor.fetchall()
                    exist = any(column[1] == 'rxntype' for column in columns)

                    if exist:
                        if isinstance(class_choice,list):
                            placeholder = ', '.join('?' for _ in class_choice)
                            query = f"SELECT * FROM {table[0]} WHERE rxntype IN ({placeholder});"
                            cursor.execute(query,class_choice)
                        else:
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE rxntype={class_choice};")
                
                    exist = any(column[1] == 'split' for column in columns)
                    if exist:
                        if split in ['train','test','val']:
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split={split};")
                        elif split == 'trainval':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('train','val');")
                        elif split == 'traintest':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('train','test');")
                        elif split == 'valtest':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('test','val');")
                        elif split == 'all':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('train','test','val');")
                     
                    rowsfound = cursor.fetchall()
                        
                    if rowsfound: self.datapath.append(table[0])
            else:
                    cursor.execute(f"PRAGMA table_info({table[0]});")
                    columns = cursor.fetchall()
                    exist = any(column[1] == 'split' for column in columns)
                    if exist:
                        if split in ['train','test','val']:
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split={split};")
                        elif split == 'trainval':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('train','val');")
                        elif split == 'traintest':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('train','test');")
                        elif split == 'valtest':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('test','val');")
                        elif split == 'all':
                            cursor.execute(f"SELECT * FROM {table[0]} WHERE split IN ('train','test','val');")
                     
                    rowsfound = cursor.fetchall()
                    if rowsfound: self.datapath.append(table[0])
                        
            conn.close()
        elif datastorage == 'MongoDB':
            client = pymongo.MongoClient('mongodb://localhost:27017/')
            database_name = self.root.split('/')[-1]
            db = client[database_name]
            collection_names = db.list_collection_names()
            self.datapath = []
            collection_names = []
            # Loop through each collection
            for collection_name in collection_names:
                # Get the collection
                collection = db[collection_name]
                if class_choice is not None:
                    # Find documents with the specified criteria
                    if split in ['train','test','val']:
                        if isinstance(class_choice,list):query = {"rxntype": {"$in": class_choice}, "split": {"$regex": split}}
                        else:query = {"rxntype": {"$regex": class_choice}, "split": {"$regex": split}}
                        documents = collection.find(query)
                        collection_names.append(collection_name)
                    elif split == 'trainval':
                        if isinstance(class_choice,list):query = {"rxntype": {"$in": class_choice}, "split":  {"$in": ['train','val']}}
                        else:query = {"rxntype": {"$regex": class_choice},  "split":  {"$in": ['train','val']}}
                        documents = collection.find(query)
                        collection_names.append(collection_name)
                    elif split == 'traintest':
                        if isinstance(class_choice,list):query = {"rxntype": {"$in": class_choice}, "split":  {"$in": ['train','test']}}
                        else:query = {"rxntype": {"$regex": class_choice}, "split":  {"$in": ['train','test']}}
                        documents = collection.find(query)
                        collection_names.append(collection_name)
                    elif split == 'valtest':
                        if isinstance(class_choice,list):query = {"rxntype": {"$in": class_choice}, "split":  {"$in": ['test','val']}}
                        else:query = {"rxntype": {"$regex": class_choice}, "split":  {"$in": ['test','val']}}
                        documents = collection.find(query)
                        collection_names.append(collection_name)
                    elif split == 'all':
                        if isinstance(class_choice,list):query = {"rxntype": {"$in": class_choice}, "split":  {"$in": ['train','test','val']}}
                        else:query = {"rxntype": {"$regex": class_choice},"split":  {"$in": ['train','test','val']}}
                        documents = collection.find(query)    
                        collection_names.append(collection_name)                    
                else:
                    if split in ['train','test','val']:
                        documents = collection.find({"split": {"$in": [split]}})
                        collection_names.append(collection_name)
                    elif split == 'trainval':
                        documents = collection.find({"split": {"$in": ['train','val']}})
                        collection_names.append(collection_name)
                    elif split == 'traintest':
                        documents = collection.find({"split": {"$in": ['train','test']}})
                        collection_names.append(collection_name)
                    elif split == 'valtest':
                        documents = collection.find({"split": {"$in": ['test','val']}})
                        collection_names.append(collection_name)
                    elif split == 'all':
                        documents = collection.find({"split": {"$in": ['train','test','val']}})
                        collection_names.append(collection_name)

                # Append documents to the result list
                self.datapath.extend(documents)
            client.close()
            


        ###### SHUFFLE THE LIST IF WE WANT TO RANDOMIZE THE DATA OR GET THE FIRST N SAMPLES
        if randomize: random.shuffle(self.datapath)
        if size is not None: random.shuffle(self.datapath)
        
        ###### GET THE FIRST N SAMPLES
        if datastorage is None:
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
                  self.datapath = self.datapath[:size]
                  self.rtypes = []
                  conn = sqlite3.connect(root.split('/')[-1]+'.db')
                  cursor = conn.cursor()
                  for table in self.datapath:
                      query = f'SELECT split FROM {table[0]};'

                      cursor.execute(query)
                      data = cursor.fetchall()

                      self.rtypes.extend(data)
            else:
                  self.rtypes = []
                  conn = sqlite3.connect(root.split('/')[-1]+'.db')
                  cursor = conn.cursor()
                  for table in self.datapath:
                      query = f'SELECT split FROM {table[0]};'

                      cursor.execute(query)
                      data = cursor.fetchall()

                      self.rtypes.extend(data)

        elif datastorage == 'MongoDB':
            if size is not None:
                  self.datapath = self.datapath[:size]
                  self.rtypes = [doc.get("rxntype") for doc in self.datapath]
            else: 
                  self.rtypes = [doc.get("rxntype") for doc in self.datapath]
                  
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
                connection = sqlite3.connect(self.root.split('/')[-1]+'.db')
                cursor = connection.cursor()

                # Fetch column names
                cursor.execute(f'PRAGMA table_info({fn[0]});')
                columns = [column[1] for column in cursor.fetchall()]

                # Fetch data
                cursor.execute(f'SELECT * FROM {fn[0]};')
                data = cursor.fetchall()

                # Transpose the data to group values by column
                transposed_data = list(zip(*data))

                # Create a dictionary where keys are column names and values are lists of column values
                info = {column: values for column, values in zip(columns, transposed_data)}
                connection.close()
            elif self.datastorage == 'MongoDB':
                # Initialize an empty dictionary to store the transformed data
                info = {}
                # Loop through each document
                for doc in self.datapath:
                    # Loop through each key-value pair in the document
                    for key, value in doc.items():
                        # Check if the key is already in the transformed_data dictionary
                        if key in info:
                            # If yes, append the value to the existing list
                            info[key].append(value)
                        else:
                            # If no, create a new list with the value
                            info[key] = [value]

            
            

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
            if not self.molecular :
                samples += [gR,gP]
            else:
                samples += [gR]

            # Add what the SMILES and Inchi keys are
            if not self.molecular :
                samples += [torch.Tensor([info['Rsmiles'],info['Psmiles'],info['Rinchi'],info['Pinchi']])]
            else:
                samples += [torch.Tensor([info['Rsmiles'],info['Rinchi']])]

            # Add what the target tensor is
            #targettensor = [gR.number_of_nodes(),gR.number_of_edges()]
            targettensor = []
            if isinstance(self.target,list):
                targettensor += [info[output] for output in self.target]
            else:
                targettensor += [info[self.target]]
            targettensor = torch.Tensor(targettensor)
            samples += [targettensor]
            
            # Add what the Additional Tensor is if needed
            if self.addtional is not None:
                additionaltensor = []
                if isinstance(self.additional,list):
                    additionaltensor += [info[output] for output in self.additional]
                else:
                    additionaltensor += [info[self.additional]]
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
    names,types,Rgraphs, Pgraphs, smiles,targets,addtionals = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    Pbatched_graph = dgl.batch(Pgraphs)
    return names,types,Rbatched_graph, Pbatched_graph, smiles,targets,addtionals

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
    names,types,Rgraphs, smiles,targets,addtionals = map(list, zip(*samples))
    Rbatched_graph = dgl.batch(Rgraphs)
    return names,types,Rbatched_graph, smiles,targets,addtionals

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
    