'''
Author: Qiyuan Zhao, Sai Mahit Vaddadi

Last Edit: November 30,2023
'''

import argparse
import os
import torch
import logging
import sys
import importlib
import shutil
import numpy as np
import joblib
from tqdm import tqdm
from dataset import RGD1Dataset,collateall,collatetargetsonly,collatewitthadditonals,collatewithaddons,molecularcollateall,molecularcollatetargetsonly,molecularcollatewitthadditonals,molecularcollatewithaddons
import hydra
import omegaconf
import pandas as pd



"""
Function that Predicts Values given the original model. 

Parameters
----------
config: string
        Path to the config file


Parameters for Config File 
----------

---- Model Initialization ----
base_model: String
        Location of the Base model to run predictions on. 
model: Variable with attribute name.
        Location of the model code to run the Predictions on.
data_path: String
        Location of the .json files with the data.
model_type: String
        Type of model being used. 

---- Dataloader ----
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

---- Prediction ----
loss: String or list
        Loss Function to use

---- Model Predictions ----
destination: string
        Location of the saved predictions. 


Returns
-------
test: .csv file
        Saved Predictions
"""







#@hydra.main(config_path='config', config_name='test')

def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    torch.set_grad_enabled(True) 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.info('GPU does not load. Cannot run.')
    logger = logging.getLogger(__name__)

    ###### LOAD MODEL WE WANT TO PREDICT ON
    try:
        checkpoint = torch.load(args.base_model)
        ###### SEE WHAT THE MODEL LOOKS LIKE 
        #predictor = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'EGAT_Rxn')(args).to(device)
        predictor = getattr(importlib.import_module('models.{}.model'.format(args.defaults.model)), 'EGAT_Rxn')(args).to(device)
        predictor.load_state_dict(checkpoint['model_state_dict'])
        
    except Exception as e:
        print(e)
        print('Error: Model Loading Failed. Please load one that works.')
        return None

    ###### LOAD PATH WE WANT TO USE
    root = hydra.utils.to_absolute_path(args.data_path)

    # LOAD THE EXCLUDE FILE
    exclude = []
    try:
        with open(args.exclude,'r') as f:
            for lc,lines in enumerate(f):
                exclude.append(lines.split('/')[-1].split('.json')[0])
    except:
        pass
    
 
    ###### SET WHAT PARTS OF THE DATASET WE ARE LOOKING AT 

    ###### LOAD DATASET TO TORCH
    TEST_DATASET = RGD1Dataset(root=root, npoints=args.npoints,split=args.split, class_choice=args.class_choice, exclude=exclude,randomize=args.randomize,fold=args.fold,foldtype=args.foldtype,size =args.size,target=args.target,additional=args.additionals,hasaddons=args.hasaddons,molecular = args.molecular,datastorage=args.datastorage)

    
    if args.hasaddons:
        if args.additionals is not None:
            if args.molecular:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollateall)
            else:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collateall)
        else:
            if args.molecular:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewithaddons)
            else:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewithaddons)
    else:
        if args.additionals is not None:
            if args.molecular:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewitthadditonals)
            else:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewitthadditonals)
        else:
            if args.molecular:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatetargetsonly)
            else:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatetargetsonly)

   
            
    ###### LOAD THE LOSS FUNCTION - This is an work in progress as we can add more functions to as time goes on. 
    lossdict = []
    if args.pred_loss == 'regular':
        criterions = [torch.nn.L1Loss(),torch.nn.MSELoss()]
        crit_list = ['MAE','RMSE']
    elif args.pred_loss == 'RMSE':
        criterions = [torch.nn.MSELoss()]
        crit_list = ['RMSE']
    elif args.pred_loss == 'MAE':
        criterions = [torch.nn.L1Loss()]
        crit_list = ['MAE']
    else:
        print('Error: Loss Function not Given')
        return None


    ###### LOAD THE PREDICTION DATAFRAME AND ITS COLUMNS
    if not args.molecular:
        test = []
        columns = ['ID','RTYPE','Rsmiles','Psmiles','Rinchi','Pinchi']
        if isinstance(args.target,list) or isinstance(args.target,omegaconf.listconfig.ListConfig):
            preds  = [t+'_PRED' for t in args.target]
            columns += preds        
            columns += args.target
        else:
            columns += [args.target+'_PRED']
            columns += [args.target]
    else:
        test = []
        columns = ['ID','RTYPE','Rsmiles','Rinchi']
        if isinstance(args.target,list) or isinstance(args.target,omegaconf.listconfig.ListConfig):
            preds  = [t+'_PRED' for t in args.target]
            columns += preds        
            columns += args.target
        else:
            columns += [args.target+'_PRED']
            columns += [args.target]

    embeddingslist = []
    if args.AttentionMaps: 
        if not args.molecular:
            RAttnMaps = []
            PAttnMaps = []
        else:
            RAttnMaps = []
    ###### MODEL EVALUATION
    with torch.no_grad():

        predictor = predictor.eval()
        for item in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
            if args.hasaddons:
                if args.additionals is not None:
                    if args.molecular: 
                        id,rtypes,Rgs,smiles,targets,additionals,Radd= item
                    else:
                        #collateall
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals,Radd,Padd = item
                else:
                    if args.model_type in ['Hr','BEP','Hr_multi']:
                        logger.info('Error: Predictions Require Additional Values that are not given.')
                        break 
                    else:
                        if args.molecular: 
                            id,rtypes,Rgs,smiles,targets,Radd= item
                        else:
                            id,rtypes,Rgs,Pgs,smiles,targets,Radd = item
                    
            else:
                if args.additionals is not None:
                    if args.molecular:
                        id,rtypes,Rgs,smiles,targets,additionals = item
                    else:
                        id,rtypes,Rgs,Pgs,smiles,targets,additionals = item
                else:
                    if args.model_type in ['Hr','BEP','Hr_multi']:
                        logger.info('Error: Predictions Require Additional Values that are not given.')
                        break 
                    else:
                        if args.molecular:
                            id,rtypes,Rgs,smiles,targets = item
                        else:
                            id,rtypes,Rgs,Pgs,smiles,targets = item

                    
            if args.model_type == 'BEP':
                ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
                if isinstance(args.additionals,list) or isinstance(args.additionals,omegaconf.listconfig.ListConfig):
                    logger.info('Error: BEP-like Prediciton can only be done on one additional set of values.')
                    break
                else:
                    Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

            elif args.model_type == 'direct':
                ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                target    = torch.Tensor([float(i[0]) for i in targets]).view(args.batch_size,1).to(device)
            
            elif args.model_type == 'Hr':
                    
                ##### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                target    = torch.Tensor([float(i[0]) for i in targets]).view(args.batch_size,1).to(device)

                ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                if isinstance(args.additionals,list) or isinstance(args.additionals,omegaconf.listconfig.ListConfig):
                    if args.Norm is not None:
                        scaler = joblib.load('scaler_model.joblib')
                        Hr        = additionals.float().view(args.batch_size,len(args.additional)).numpy()
                        Hr        = scaler.transform(Hr).to(device)

                    else:
                        Hr        = additionals.float().view(args.batch_size,len(args.additional)).to(device)
                else:
                    if args.Norm is not None:
                        Hr        = additionals.float().view(args.batch_size,1).numpy()
                        Hr        = scaler.transform(Hr).to(device)
                        
                    else:
                        Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

            elif args.model_type == 'multi':
                ###### LOAD ALL TARGETS TO CUDA
                if isinstance( args.target,list) or isinstance(args.target,omegaconf.listconfig.ListConfig):
                    target = targets.float().view(args.batch_size,len(target)).to(device)
                else:
                    logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                    break 

            elif args.model_type == 'Hr_multi':
                 ###### LOAD ALL TARGETS TO CUDA
                if isinstance( args.target,list) or isinstance(args.target,omegaconf.listconfig.ListConfig):
                    target = targets.float().view(args.batch_size,len(target)).to(device)
                else:
                    logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                    break 
                ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                if isinstance(args.additionals,list) or isinstance(args.additionals,omegaconf.listconfig.ListConfig):
                    Hr        = additionals.float().view(args.batch_size,len(args.additionals)).to(device)
                else:
                    Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

            ##### TAKE THE GRAPHS AND LOAD THEM INTO CUDA
            if not args.molecular:
                RGgs      = Rgs.to(device)
                PGgs      = Pgs.to(device)
                RGgs.ndata['x'].to(device)
                RGgs.edata['x'].to(device)
                PGgs.ndata['x'].to(device)
                PGgs.edata['x'].to(device)
                if args.hasaddons: 
                    RAdd = Radd.to(device)
                    PAdd = Padd.to(device)
            else:
                RGgs      = Rgs.to(device)
                RGgs.ndata['x'].to(device)
                RGgs.edata['x'].to(device)
                if args.hasaddons: 
                    RAdd = Radd.to(device)

                    
            
            if args.model_type == 'direct':
                ###### GET PREDICTION
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs, PGgs)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs, PGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs)
                            else:
                                pred,Rmap = predictor(RGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,RAdd)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs)
                                
                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                if args.Embed == 0 or args.Embed == 1: 
                    batch_result = torch.cat([pred,target],1).cpu().data
                    batch_result = np.concatenate((np.array(smiles),batch_result.numpy()),axis=1)
                    id = np.array(id).reshape(-1,1)
                    rtypes = np.array(rtypes).reshape(-1,1)
                    
                    batch_result = np.concatenate((id,rtypes,batch_result),axis=1)
                    if args.Embed == 1: 
                        embeddingslist.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                    
                elif args.Embed == 2:
                    embeddingslist.append(embeddings.cpu().numpy().tolist())
                
                if args.AttentionMaps:
                    if not args.molecular:  
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]
                        PAttnMaps += [str(_) for _ in Pmap.cpu().numpy().tolist()]
                    else:
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]
                        


            elif args.model_type == 'BEP':
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                outputs = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                outputs,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                        else:
                            if not args.AttentionMaps:
                                outputs = predictor(RGgs, PGgs)
                            else:
                                outputs,Rmap = predictor(RGgs, PGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                outputs,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                outputs,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                        else:
                            if not args.AttentionMaps:
                                outputs,embeddings = predictor(RGgs, PGgs)
                            else:
                                outputs,embeddings,Rmap = predictor(RGgs, PGgs)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                outputs = predictor(RGgs,RAdd)
                            else:
                                outputs,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttentionMaps:
                                outputs = predictor(RGgs)
                            else:
                                outputs,Rmap = predictor(RGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                outputs,embeddings = predictor(RGgs,RAdd)
                            else:
                                outputs,embeddings,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttentionMaps:
                                outputs,embeddings = predictor(RGgs)
                            else:
                                outputs,embeddings,Rmap = predictor(RGgs)
                    
                        
                        
                ###### GET PREDICTION
                if args.Embed == 0 or args.Embed == 1: 
                    pred = outputs[:, 0].unsqueeze(1) * Hr + outputs[:, 1].unsqueeze(1)

                    batch_result = torch.cat([pred,target],1).cpu().data
                    batch_result = np.concatenate((np.array(smiles),batch_result.numpy()),axis=1)
                    id = np.array(id).reshape(-1,1)
                    rtypes = np.array(rtypes).reshape(-1,1)
                    
                    batch_result = np.concatenate((id,rtypes,batch_result),axis=1)
                    if args.Embed == 1: 
                        embeddingslist.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                
                elif args.Embed == 2:
                    embeddingslist.append(embeddings.cpu().numpy().tolist())
                
                if args.AttentionMaps:
                    if not args.molecular:  
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]
                        PAttnMaps += [str(_) for _ in Pmap.cpu().numpy().tolist()]
                    else:
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]
                
            elif args.model_type == 'Hr':

                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr)

            

                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                if args.Embed == 0 or args.Embed == 1: 
                    batch_result = torch.cat([pred,target],1).cpu().data
                    batch_result = np.concatenate((np.array(smiles),batch_result.numpy()),axis=1)
                    id = np.array(id).reshape(-1,1)
                    rtypes = np.array(rtypes).reshape(-1,1)
                    
                    batch_result = np.concatenate((id,rtypes,batch_result),axis=1)
                    if args.Embed == 1: 
                        embeddingslist.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                    
                elif args.Embed == 2:
                    embeddingslist.append(embeddings.cpu().numpy().tolist())
                
                if args.AttentionMaps:
                    if not args.molecular:
                        RAttnMaps += [str(Rmap.cpu().numpy().tolist())]
                        PAttnMaps += [str(Pmap.cpu().numpy().tolist())]
                    else:
                        RAttnMaps += [str(Rmap.cpu().numpy().tolist())]

            elif args.model_type == 'Hr_multi':
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr)
 

            

                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                if args.Embed == 0 or args.Embed == 1: 
                    # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                    batch_result = torch.cat([pred,target],1).cpu().data
                    batch_result = np.concatenate((np.array(smiles),batch_result.numpy()),axis=1)
                    id = np.array(id).reshape(-1,1)
                    rtypes = np.array(rtypes).reshape(-1,1)
                    
                    batch_result = np.concatenate((id,rtypes,batch_result),axis=1)
                    if args.Embed == 1: 
                        embeddingslist.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                    
                elif args.Embed == 2:
                    embeddingslist.append(embeddings.cpu().numpy().tolist())
                
                if args.AttentionMaps:
                    if not args.molecular:  
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]
                        PAttnMaps += [str(_) for _ in Pmap.cpu().numpy().tolist()]
                    else:
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]

            elif args.model_type == 'multi':
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs, PGgs)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs, PGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred = predictor(RGgs)
                            else:
                                pred,Rmap = predictor(RGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs,RAdd)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttentionMaps:
                                pred,embeddings = predictor(RGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs)
                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                if args.Embed == 0 or args.Embed == 1: 
                    # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                    batch_result = torch.cat([pred,target],1).cpu().data
                    batch_result = np.concatenate((np.array(smiles),batch_result.numpy()),axis=1)
                    id = np.array(id).reshape(-1,1)
                    rtypes = np.array(rtypes).reshape(-1,1)
                    
                    batch_result = np.concatenate((id,rtypes,batch_result),axis=1)
                    if args.Embed == 1: 
                        embeddingslist.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                    
                elif args.Embed == 2:
                    embeddingslist.append(embeddings.cpu().numpy().tolist())
                
                if args.AttentionMaps:
                    if not args.molecular:  
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]
                        PAttnMaps += [str(_) for _ in Pmap.cpu().numpy().tolist()]
                    else:
                        RAttnMaps += [str(_) for _ in Rmap.cpu().numpy().tolist()]

            if args.Embed == 0 or args.Embed == 1:
                ####### GET BATCH LOSSES
                batch = []
                j = 0
                multi_columns = []
                for criterion in criterions:
                    # Check if it is multitask or not 
                    if 'multi' not in args.model_type:
                        batch += [criterion(pred, target).cpu().data.numpy()]
                    else:
                        for i in range(len(args.target)):
                            batch += [criterion(pred[:,i].unsqueeze(1), target[:,i].unsqueeze(1)).cpu().data.numpy()]
                            multi_columns += [f'{crit_list[j]}_{args.target[i]}']
                    
                    if len(criterions) > 1:
                        j += 1

                lossdict.append(batch)
            
     
    ###### SAVE TESTS AS CSV
    if args.Embed == 0:
        test = np.concatenate(tuple(test))
        test = pd.DataFrame(test,columns=columns)
        test.to_csv(args.destination)
    
    elif args.Embed == 1:
        test = np.concatenate(tuple(test))
        test = pd.DataFrame(test,columns=columns)
        test.to_csv(args.destination)

        embeddingslist = np.concatenate(tuple(embeddingslist))
        testembeddings = pd.DataFrame(embeddingslist)
        testembeddings.to_csv(args.destination[:-4]+'_embeddings.csv')
    
    elif args.Embed == 2:
        testembeddings = pd.DataFrame(embeddingslist)
        testembeddings.to_csv(args.destination[:-4]+'_embeddings.csv')
    
    if args.AttentionMaps:
        if args.molecular:
            AttnMaps = pd.DataFrame({'React':RAttnMaps,'Prod':PAttnMaps})
        else:
            AttnMaps = pd.DataFrame({'React':RAttnMaps})
        AttnMaps.to_csv(args.destination[:-4]+'_amap.csv')


            
    
    if args.Embed == 0 or args.Embed == 1:
        print(np.mean(lossdict))
        if 'multi' not in args.model_type:
            losses = pd.DataFrame(lossdict,columns=crit_list)
        else:
            if args.pred_loss in ['MAE','RMSE']:
                logger.info(f'Test {args.pred_loss}: {np.round(np.mean(lossdict),3)} kcal/mol')
            else:
                logger.info(f'Test MAE: {np.round(np.mean(lossdict),3)} kcal/mol')

            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--config", type=str, default="config/test.yaml", help="Path to the config file")

    args = parser.parse_args()

    # Load the specified config file
    config = omegaconf.OmegaConf.load(args.config)
    omegaconf.OmegaConf.set_struct(config, False)
    
    # Determine the config_name based on the name of the loaded config file
    file_name = os.path.basename(args.config)
    config_name, _ = os.path.splitext(file_name)

    # Set the config_name for the Hydra function
    #hydra.utils.set_config_name(config_name)

    # Run the Hydra function with the merged configuration
    main(config)
