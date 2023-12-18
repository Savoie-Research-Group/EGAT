from sqlite3 import SQLITE_ALTER_TABLE
import sqlite3
import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import omegaconf
import hydra
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import numpy as np
import pymongo

"""
Function that Splits the .json given the original model. 

Parameters
----------
config: string
        Path to the config file


Parameters for Config File 
----------

---- Model Initialization ----
data_path: String
        Location of the .json files with the data.
test_split: float
        % of data in Testing and Validation set. 
test_only: bool 
        Check if we want an external testing set. 
folds: int 
        Number of Folds being made
state: int 
        Random Seed.

Returns
-------
Folder with saved locations
"""



def Split(args):
    omegaconf.OmegaConf.set_struct(args, False)

    if args.datastorage is None:
        # generate the first split
        fs = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_path) for f in filenames if (fnmatch.fnmatch(f,"*.json") )])
        fs = ['/'.join(i.split('/')[1:]).split('.')[0] for i in fs]

        train, valtest, _, _ = train_test_split(fs,fs,test_size=args.test_split,random_state=args.state)
        val, test, _, _ = train_test_split(valtest,valtest,test_size=0.5,random_state=args.state)

        if args.folds == 1:
            ### generate    
            with open(os.path.join(args.data_path,'train_test_split/train_idx.json'),'w') as f:
                json.dump(train,f)

            with open(os.path.join(args.data_path,'train_test_split/val_idx.json'),'w') as f:
                json.dump(val,f)

            with open(os.path.join(args.data_path,'train_test_split/test_idx.json'),'w') as f:
                json.dump(test,f)

        else:
        
            # Take the training and validation and run x-fold cross validation. 
            kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.state)

            combined_train_val = np.concatenate((train, val))

            # Perform k-fold cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(combined_train_val), 1):
                # Split the data into training and validation sets for this fold
                fold_train_data, fold_val_data = combined_train_val[train_idx], combined_train_val[val_idx]
                if not os.path.isdir(os.path.join(args.data_path,'train_test_split',f'fold{fold_idx}')): os.mkdir(os.path.join(args.data_path,'train_test_split',f'fold{fold_idx}'))


                with open(os.path.join(args.data_path,f'train_test_split/fold{fold_idx}/train_idx.json'),'w') as f:
                    json.dump(train_idx,f)
                
                with open(os.path.join(args.data_path,f'train_test_split/fold{fold_idx}/val_idx.json'),'w') as f:
                    json.dump(val_idx,f)

                with open(os.path.join(args.data_path,f'train_test_split/fold{fold_idx}/test_idx.json'),'w') as f:
                    json.dump(test,f)
    
    elif args.datastorage == 'SQL':
        conn = sqlite3.connect(args.data_path.split('/')[-1]+'.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()

        if args.folds == 1:
            train, valtest, _, _ = train_test_split(tables,tables,test_size=args.test_split,random_state=args.state)
            val, test, _, _ = train_test_split(valtest,valtest,test_size=0.5,random_state=args.state)
            
            conn = sqlite3.connect(args.data_path.split('/')[-1]+'.db')
            cursor = conn.cursor()
            sets = ['train','val','test']
            for i,names in enumerate([train,val,test]):
                for table in names:
                    cursor.exceute(f"ALTER TABLE {[table[0]]} ADD COLUMN split TEXT;")
                    cursor.exceute(f"UPDATE {[table[0]]} SET split={sets[i]};")
                    
            conn.commit()
            conn.close()
        else:
            print('currently under construction')
            kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.state)
            combined_train_val = np.concatenate((train, val))
            conn = sqlite3.connect(args.data_path.split('/')[-1]+'.db')
            cursor = conn.cursor()
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(combined_train_val), 1):
                
                sets = ['train','val','test']
                for i,names in enumerate([train,val,test]):
                    for table in names:
                        cursor.exceute(f"ALTER TABLE {[table[0]]} ADD COLUMN fold{fold_idx}_split TEXT;")
                        cursor.exceute(f"UPDATE {[table[0]]} SET fold{fold_idx}_split={sets[i]};")
                        
            conn.commit()
            conn.close()
            


    elif args.datastorage == 'MongoDB':
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        database_name = args.data_path.split('/')[-1]
        db = client[database_name]
        collection_names = db.list_collection_names()

        train, valtest, _, _ = train_test_split(collection_names,collection_names,test_size=args.test_split,random_state=args.state)
        val, test, _, _ = train_test_split(valtest,valtest,test_size=0.5,random_state=args.state)
        if args.folds == 1:
            sets = ['train','val','test']
            for i,names in enumerate([train,val,test]):
                dsetdict = {'split':sets[i]}
                for collection_name in names:
                    db[collection_name].update_many({},{"$set":dsetdict})
            
            client.close()
        else:
            print('currently under construction')
            kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.state)

            combined_train_val = np.concatenate((train, val))
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(combined_train_val), 1):
                sets = ['train','val','test']
                for i,names in enumerate([train,val,test]):
                    dsetdict = {f'fold{fold_idx}_split':sets[i]}
                    for collection_name in names:
                        db[collection_name].update_many({},{"$set":dsetdict})
            client.close()
                

            






                    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--config", type=str, default="config/test.yaml", help="Path to the config file")

    args = parser.parse_args()

    # Load the specified config file
    config = omegaconf.OmegaConf.load(args.config)
    omegaconf.OmegaConf.set_struct(args, False)
    
    # Determine the config_name based on the name of the loaded config file
    file_name = os.path.basename(args.config)
    config_name, _ = os.path.splitext(file_name)

    # Set the config_name for the Hydra function
    hydra.utils.set_config_name(config_name)

    # Run the Hydra function with the merged configuration
    Split(config)
