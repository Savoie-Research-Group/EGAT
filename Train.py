"""
Author: Qiyuan Zhao, Sai Mahit Vaddadi

Creation Date: Feb 2023
Edit Date: December 2023

"""

### IMPORT LIBRARIES
import argparse
import os
import torch
import logging
import sys
import importlib
import shutil
import numpy as np
from tqdm import tqdm
import hydra
import omegaconf
import pandas as pd 
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import joblib

### IMPORT FUNCTIONS
from dataset import RGD1Dataset,collateall,collatetargetsonly,collatewitthadditonals,collatewithaddons,molecularcollateall,molecularcollatetargetsonly,molecularcollatewitthadditonals,molecularcollatewithaddons


### SET HYDRA to the config directory
@hydra.main(config_path='config', config_name='default')





def main(args):

    '''INITALIIZATION of PACKAGES'''

    ### Check if weights and biases is needed for live model monitoring
    if args.weightsandbiases:
        import wandb
        wandb.init(project=args.wandbproject,name=args.wandbname)
    
    ### Set Logger
    logger = logging.getLogger(__name__)


    ### Get the gradient enabled
    torch.set_grad_enabled(True)

    ### Set the GPU if available. Otherwise Stop. 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.info('GPU does not load. Cannot run.')
        return None 

    '''INITALIIZATION of MODEL'''

    ### SET THE START POINT. 
    if args.startpoint == 'restart':
        # Attempt to find a restart point. If not, start from scratch.
        try:
            checkpoint = torch.load('best_model.pth') # Set model to path
            start_epoch = checkpoint['epoch'] # Get starting point 
            logger.info('Use pretrain model, starting from epoch {}'.format(start_epoch))
            best_loss = checkpoint['test_acc'] # Get best loss for reference
        except:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # Get the gpus
            logger.info('No existing model, starting training from scratch...')
            best_loss = 100
            start_epoch = 0
    
    elif args.startpoint == 'Pretrain':
         # Attempt to find a restart point. If not, stop.
        try:
            checkpoint = torch.load(args.base_model)  # Set model to path
            logger.info('Pre-trained Model is loaded, starting training on current data from scratch...')
            best_loss = 100
            start_epoch = 0
        except:
            logger.info('Pre-trained Model Failed to Load.')
            return None
    
    else:
        # Attempt to start from scratch.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # Get the gpus
        logger.info('No existing model, starting training from scratch...')
        best_loss = 100
        start_epoch = 0

    '''INITALIIZATION of DATA'''
    ### Get the data path, otherwise stop if it's not found. 
    try:
        root = hydra.utils.to_absolute_path(args.data_path) # Get the data path
    except:
        logger.info('Data Path cannot be found')
        return None

    ### Get the files to exclude. If not, just set it blank. 
    if args.exclude is not None:
        if os.path.isfile(os.path.join(args.data_path,args.exclude)): # Check if path exists
            exclude = []
            with open(os.path.join(args.data_path,args.exclude),'r') as f: # open file 
                for lc,lines in enumerate(f):
                    exclude.append(lines.split('/')[-1].split('.json')[0]) # Add exclude files 
        else:
            logger.info(f'{os.path.join(args.data_path,args.exclude)} does not exist.')
            exclude = []
    else:
        logger.info(f'{os.path.join(args.data_path)} has not exclude file.')
        exclude = []

    '''LOADING of DATA to PYTORCH'''
    ### Create a dataset
    TRAIN_DATASET  = RGD1Dataset(root=root,split='train', class_choice=args.class_choice, exclude=exclude,randomize=args.randomize,fold=args.fold,foldtype=args.foldtype,size=args.size,target=args.target,additional=args.additionals,hasaddons=args.addons,molecular=args.molecular)
    TEST_DATASET = RGD1Dataset(root=root, split='val', class_choice=args.class_choice, exclude=exclude,randomize=args.randomize,fold=args.fold,foldtype=args.foldtype,size=args.size,target=args.target,additional=args.additionals,hasaddons=args.addons,molecular=args.molecular)
    if not args.test_only: EXT_DATASET = RGD1Dataset(root=root, split='test', class_choice=args.class_choice, exclude=exclude,randomize=args.randomize,fold=args.fold,foldtype=args.foldtype,size=args.size,target=args.target,additional=args.additionals,hasaddons=args.addons,molecular=args.molecular)
    
    ### Create a torch.DataLoader
    if args.hasaddons: #Check if we need RDKit Global Features. If we do, load them.
        if args.additionals is not None: # Check if there are added features. If we do, load them.
            if args.molecular: # Check if we only need molecular features. If we do, only load R features. 
                # Load data for test/validation set. 
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollateall)
                # Load Training Data
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollateall)
                # Only loads the external test set if we set that explicitly.
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollateall) 
            else:
                # Load data for test/validation set. 
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collateall)
                # Load Training Data
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collateall)
                # Only loads the external test set if we set that explicitly.
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collateall)
        else:
            if args.molecular:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewithaddons)
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewithaddons)
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewithaddons)
            else:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewithaddons)
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewithaddons)
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewithaddons)
    else:
        if args.additionals is not None:
            if args.molecular:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewitthadditonals)
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewitthadditonals)
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatewitthadditonals)
            else:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewitthadditonals)
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewitthadditonals)
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatewitthadditonals)
        else:
            if args.molecular:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatetargetsonly)
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatetargetsonly)
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=molecularcollatetargetsonly)
            else:
                testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatetargetsonly)
                trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatetargetsonly)
                if not args.test_only: extDataLoader = torch.utils.data.DataLoader(EXT_DATASET, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, drop_last=args.drop_last, collate_fn=collatetargetsonly)


    '''MODEL LOADING'''
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    predictor = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'EGAT_Rxn')(args).to(device)
    print(predictor)

    # count the total number of parameters in the model
    total_params = sum(p.numel() for p in predictor.parameters())

    # count the number of trainable parameters in the model
    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    try:
        predictor.load_state_dict(checkpoint['model_state_dict'])
        print("Update predictor")
    except:
        print('Cannot load State Dict. Stopping Run.')
        return None 
        #pass
    
    '''SET LOSS FUNCTION'''
    lossdict = []
    if args.train_loss == 'RMSE':
        criterion = torch.nn.MSELoss()
    elif args.train_loss == 'MAE':
        criterion = torch.nn.L1Loss()
    else:
        print('Error: Loss Function not Given')
        return None

    '''SET OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=args.learning_rate,
            betas=tuple(args.betas), #(0.9, 0.999)
            eps=args.epsilon,#1e-08
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(predictor.parameters(), lr=args.learning_rate, momentum=args.momentum) #0.9

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = args.learning_rate_min
    MOMENTUM_ORIGINAL = args.momentum_orig #.1
    MOMENTUM_DECAY = args.lr_decay
    MOMENTUM_DECAY_STEP = args.step_size

    global_epoch = 0
    loss_increase_count = 0

    if args.Norm is not None:
        if args.model_type in ['Hr','Hr_multi']:
            TrainHr = np.array([])
            for item in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
                if args.hasaddons:
                    if args.additionals is not None:
                        if args.molecular: 
                            id,rtypes,Rgs,smiles,targets,additionals,Radd= item
                        else:
                            id,rtypes,Rgs,Pgs,smiles,targets,additionals,Radd,Padd = item        
                else:
                    if args.additionals is not None:
                        if args.molecular:
                            id,rtypes,Rgs,smiles,targets,additionals = item
                        else:
                            id,rtypes,Rgs,Pgs,smiles,targets,additionals = item
                
                if isinstance(args.additionals,list):
                    Hr        = additionals.float().view(args.batch_size,len(args.additionals)).cpu().numpy()
                else:
                    Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).cpu().numpy()

                if TrainHr == []:
                    TrainHr = Hr
                else:
                    TrainHr = np.concatenate((TrainHr,Hr),axis=0)
                
                del id
                del rtypes
                del Rgs
                del smiles
                del targets
                if args.additionals is not None:del additionals
                if args.hasaddons: del Radd

                if not args.molecular:
                    del Pgs
                    if args.hasaddons: del Padd
                    
                
        if isinstance(args.additionals,list):
            TrainHr = TrainHr.reshape(-1,len(args.additional))
        else:
            TrainHr = TrainHr.reshape(-1,1)

        scaler = StandardScaler()
        scaler.fit(TrainHr)
        joblib.dump(scaler, 'scaler.joblib')


    ###### LOAD THE PREDICTION DATAFRAME AND ITS COLUMNS
    
    for epoch in range(start_epoch, args.epoch+args.epoch_const):
        loss_increase_count += 1
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        '''Adjust learning rate and BN momentum'''
        # set up lr
        if epoch < args.epoch:
            if args.scheduler == 'step':
                lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
            elif args.scheduler == 'exp':
                if epoch < args.warmup:
                    lr_factor = (epoch + 1) * 1.0 / args.warmup
                else:
                    lr_factor = np.exp( - args.expdecay * (epoch - args.warmup + 1) / args.epoch)
                lr = max(args.learning_rate * lr_factor, LEARNING_RATE_CLIP)
            elif args.scheduler == 'cos':
                if epoch < args.warmup:
                    lr = args.learning_rate * (epoch + 1) * 1.0 / args.warmup
                else:
                    lr = LEARNING_RATE_CLIP + 0.5 * (args.learning_rate - LEARNING_RATE_CLIP) * (1 + np.cos(np.pi * (epoch - args.warmup + 1) / (args.epoch - args.warmup + 1)))
        else:
            lr = LEARNING_RATE_CLIP

        # update lr
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        predictor = predictor.apply(lambda x: bn_momentum_adjust(x, momentum))
        predictor = predictor.train()

        '''learning one epoch'''
        ###### LOAD THE PREDICTION DATAFRAME AND ITS COLUMNS
        if not args.molecular:
            train = []
            columns = ['ID','RTYPE','Rsmiles','Psmiles','Rinchi','Pinchi']
            if isinstance(args.target,list):
                preds  = [t+'_PRED' for t in args.target]
                columns += preds        
                columns += args.target
            else:
                columns += [args.target+'_PRED']
                columns += [args.target]
        else:
            train = []
            columns = ['ID','RTYPE','Rsmiles','Rinchi']
            if isinstance(args.target,list):
                preds  = [t+'_PRED' for t in args.target]
                columns += preds        
                columns += args.target
            else:
                columns += [args.target+'_PRED']
                columns += [args.target]


        loss_list = []
        training_embeddings = []
        for item in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
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
                            id,rtypes,Rgs,smiles,targets,Radd = item
                    
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
                            id,rtypes,Rgs,Pgs,smiles,targets = item
                        else:
                            id,rtypes,Rgs,smiles,targets = item
                    
        
            if args.model_type == 'BEP':
                ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
                if isinstance(args.additionals,list):
                    logger.info('Error: BEP-like Prediciton can only be done on one additional set of values.')
                    break
                else:
                    Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

            elif args.model_type == 'direct':
                ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
            
            elif args.model_type == 'Hr':
                ##### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)

                ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                if isinstance(args.additionals,list):
                    if args.Norm is not None:
                        Hr        = additionals.float().view(args.batch_size,len(args.additionals)).numpy()
                        Hr        = scaler.transform(Hr).to(device)

                    else:
                        Hr        = additionals.float().view(args.batch_size,len(args.additionals)).to(device)
                else:
                    if args.Norm is not None:
                        Hr        = additionals.float().view(args.batch_size,1).numpy()
                        Hr        = scaler.transform(Hr).to(device)
                        
                    else:
                        Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

            elif args.model_type == 'multi':
                ###### LOAD ALL TARGETS TO CUDA
                if isinstance(args.target,list):
                    target = targets.float().view(args.batch_size,len(args.target)).to(device)
                else:
                    logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                    break 

            elif args.model_type == 'Hr_multi':
                    ###### LOAD ALL TARGETS TO CUDA
                if isinstance(args.target,list):
                    target = targets.float().view(args.batch_size,len(args.target)).to(device)
                else:
                    logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                    break 
                ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                if isinstance(args.additionals,list):
                    if args.Norm is not None:
                        Hr        = additionals.float().view(args.batch_size,len(args.additionals)).numpy()
                        Hr        = scaler.transform(Hr).to(device)

                    else:
                        Hr        = additionals.float().view(args.batch_size,len(args.additionals)).to(device)
                else:
                    if args.Norm is not None:
                        Hr        = additionals.float().view(args.batch_size,1).numpy()
                        Hr        = scaler.transform(Hr).to(device)
                        
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

            optimizer.zero_grad()
            if args.model_type == 'direct':
                ###### GET PREDICTION
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs, PGgs)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = pred = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = pred = predictor(RGgs,PGgs,RAdd,PAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs, PGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs)
                            else:
                                pred,Rmap = predictor(RGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = pred = predictor(RGgs,RAdd)
                            else:
                                pred,embeddings,Rmap = pred = predictor(RGgs,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs)
                            
                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                batch_result = torch.cat([pred,target]).cpu().data
                batch_result = torch.cat([smiles,batch_result]).numpy()
                batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                if args.Embed == 1: training_embeddings.append(embeddings.cpu().numpy().tolist())
                train.append(batch_result)

                loss = criterion(pred, target)

            elif args.model_type == 'BEP':
                ###### GET PREDICTION
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                outputs = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                outputs,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                        else:
                            if not args.AttnMaps:
                                outputs = predictor(RGgs, PGgs)
                            else:
                                outputs,Rmap = predictor(RGgs, PGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                outputs,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                outputs,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                        else:
                            if not args.AttnMaps:
                                outputs,embeddings = predictor(RGgs, PGgs)
                            else:
                                outputs,embeddings,Rmap = predictor(RGgs, PGgs)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                outputs = predictor(RGgs,RAdd)
                            else:
                                outputs,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttnMaps:
                                outputs = predictor(RGgs)
                            else:
                                outputs,Rmap = predictor(RGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                outputs,embeddings = predictor(RGgs,RAdd)
                            else:
                                outputs,embeddings,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttnMaps:
                                outputs,embeddings = predictor(RGgs)
                            else:
                                outputs,embeddings,Rmap = predictor(RGgs)
                    
                pred = outputs[:, 0].unsqueeze(1) * Hr + outputs[:, 1].unsqueeze(1)
                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                batch_result = torch.cat([pred,target]).cpu().data
                batch_result = torch.cat([smiles,batch_result]).numpy()
                batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                if args.Embed == 1: training_embeddings.append(embeddings.cpu().numpy().tolist())
                train.append(batch_result)

                loss = criterion(pred, target)
            
            elif args.model_type == 'Hr':
                
                ###### GET PREDICTION
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr)
                
                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                batch_result = torch.cat([pred,target]).cpu().data
                batch_result = torch.cat([smiles,batch_result]).numpy()
                batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                if args.Embed == 1: training_embeddings.append(embeddings.cpu().numpy().tolist())
                train.append(batch_result)

                loss = criterion(pred, target)
                
            elif args.model_type == 'Hr_multi':
                ###### GET PREDICTION
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs, PGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,Hr)
                            else:
                                pred,Rmap = predictor(RGgs,Hr)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,Hr,RAdd)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,Hr)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,Hr)
                
                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                batch_result = torch.cat([pred,target]).cpu().data
                batch_result = torch.cat([smiles,batch_result]).numpy()
                batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                if args.Embed == 1: training_embeddings.append(embeddings.cpu().numpy().tolist())
                train.append(batch_result)

                if len(args.tweights) == len(args.targets):
                    loss = None
                    for index,weight in enumerate(args.tweights):
                        p = pred[:,index].unsqueeze(1)
                        t = target[:,index].unsqueeze(1)
                        if loss is not None:
                            loss += args.tweights[index] * criterion(p,t)
                        else:
                            loss = args.tweights[index] * criterion(p,t)
                else:
                    print('Cannot work becauze the weights are underdetermined.')
                    return None 

            elif args.model_type == 'multi':
                ###### GET PREDICTION
                if not args.molecular:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs, PGgs)
                            else:
                                pred,Rmap = predictor(RGgs, PGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs, PGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs, PGgs)

                else:
                    if args.Embed == 0:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred = predictor(RGgs,RAdd)
                            else:
                                pred,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred = predictor(RGgs)
                            else:
                                pred,Rmap = predictor(RGgs)
                    elif args.Embed == 1:
                        if args.additionals is not None:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs,RAdd)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs,RAdd)
                        else:
                            if not args.AttnMaps:
                                pred,embeddings = predictor(RGgs)
                            else:
                                pred,embeddings,Rmap = predictor(RGgs)
                
                # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                batch_result = torch.cat([pred,target]).cpu().data
                batch_result = torch.cat([smiles,batch_result]).numpy()
                batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                if args.Embed == 1: training_embeddings.append(embeddings.cpu().numpy().tolist())
                train.append(batch_result)

                if len(args.tweights) == len(args.targets):
                    loss = None
                    for index,weight in enumerate(args.tweights):
                        p = pred[:,index].unsqueeze(1)
                        t = target[:,index].unsqueeze(1)
                        if loss is not None:
                            loss += args.tweights[index] * criterion(p,t)
                        else:
                            loss = args.tweights[index] * criterion(p,t)
                else:
                    print('Cannot work becauze the weights are underdetermined.')
                    return None 
 
            loss.backward()
            loss_list.append(loss.cpu().data.numpy())
            optimizer.step()

        train_instance_acc = np.mean(loss_list)
        logger.info('Train accuracy is: %.5f' % train_instance_acc)
              
        # model evaluation
        with torch.no_grad():

            predictor = predictor.eval()
            
            if not args.molecular:
                test = []
                columns = ['ID','RTYPE','Rsmiles','Psmiles','Rinchi','Pinchi']
                if isinstance(args.target,list):
                    preds  = [t+'_PRED' for t in args.target]
                    columns += preds        
                    columns += args.target
                else:
                    columns += [args.target+'_PRED']
                    columns += [args.target]
            else:
                test = []
                columns = ['ID','RTYPE','Rsmiles','Rinchi']
                if isinstance(args.target,list):
                    preds  = [t+'_PRED' for t in args.target]
                    columns += preds        
                    columns += args.target
                else:
                    columns += [args.target+'_PRED']
                    columns += [args.target]


            val_loss = []
            validation_embeddings = []
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
                                id,rtypes,Rgs,smiles,targets,Radd = item       
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
                                id,rtypes,Rgs,Pgs,smiles,targets = item
                            else:
                                id,rtypes,Rgs,smiles,targets = item
                        
            
                if args.model_type == 'BEP':
                    ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                    target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
                    if isinstance(args.additionals,list):
                        logger.info('Error: BEP-like Prediciton can only be done on one additional set of values.')
                        break
                    else:
                        Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

                elif args.model_type == 'direct':
                    ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                    target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
                
                elif args.model_type == 'Hr':
                    ##### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                    target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)

                    ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                    if isinstance(args.additionals,list):
                        if args.Norm is not None:
                            Hr        = additionals.float().view(args.batch_size,len(args.additionals)).numpy()
                            Hr        = scaler.transform(Hr).to(device)

                        else:
                            Hr        = additionals.float().view(args.batch_size,len(args.additionals)).to(device)
                    else:
                        if args.Norm is not None:
                            Hr        = additionals.float().view(args.batch_size,1).numpy()
                            Hr        = scaler.transform(Hr).to(device)
                            
                        else:
                            Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)


                elif args.model_type == 'multi':
                    ###### LOAD ALL TARGETS TO CUDA
                    if isinstance(args.target,list):
                        target = targets.float().view(args.batch_size,len(args.target)).to(device)
                    else:
                        logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                        break 

                elif args.model_type == 'Hr_multi':
                        ###### LOAD ALL TARGETS TO CUDA
                    if isinstance(args.target,list):
                        target = targets.float().view(args.batch_size,len(args.target)).to(device)
                    else:
                        logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                        break 
                    ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                    if isinstance(args.additionals,list):
                        if args.Norm is not None:
                            Hr        = additionals.float().view(args.batch_size,len(args.additionals)).numpy()
                            Hr        = scaler.transform(Hr).to(device)

                        else:
                            Hr        = additionals.float().view(args.batch_size,len(args.additionals)).to(device)
                    else:
                        if args.Norm is not None:
                            Hr        = additionals.float().view(args.batch_size,1).numpy()
                            Hr        = scaler.transform(Hr).to(device)
                            
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
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs, PGgs)
                                else:
                                    pred,Rmap = predictor(RGgs, PGgs)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs, PGgs)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs, PGgs)

                    else:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,RAdd)
                                else:
                                    pred,Rmap = predictor(RGgs,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs)
                                else:
                                    pred,Rmap = predictor(RGgs)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,RAdd)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs)
                
                    # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                    batch_result = torch.cat([pred,target]).cpu().data
                    batch_result = torch.cat([smiles,batch_result]).numpy()
                    batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                    if args.Embed == 1: validation_embeddings.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                    loss = criterion(pred, target)
                
                elif args.model_type == 'BEP':
                    ###### GET PREDICTION
                    if not args.molecular:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    outputs = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    outputs,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                            else:
                                if not args.AttnMaps:
                                    outputs = predictor(RGgs, PGgs)
                                else:
                                    outputs,Rmap = predictor(RGgs, PGgs)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    outputs,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    outputs,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                if not args.AttnMaps:
                                    outputs,embeddings = predictor(RGgs, PGgs)
                                else:
                                    outputs,embeddings,Rmap = predictor(RGgs, PGgs)

                    else:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    outputs = predictor(RGgs,RAdd)
                                else:
                                    outputs,Rmap = predictor(RGgs,RAdd)
                            else:
                                if not args.AttnMaps:
                                    outputs = predictor(RGgs)
                                else:
                                    outputs,Rmap = predictor(RGgs)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    outputs,embeddings = predictor(RGgs,RAdd)
                                else:
                                    outputs,embeddings,Rmap = predictor(RGgs,RAdd)
                            else:
                                if not args.AttnMaps:
                                    outputs,embeddings = predictor(RGgs)
                                else:
                                    outputs,embeddings,Rmap = predictor(RGgs)
                
                    pred = outputs[:, 0].unsqueeze(1) * Hr + outputs[:, 1].unsqueeze(1)
                    # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                    batch_result = torch.cat([pred,target]).cpu().data
                    batch_result = torch.cat([smiles,batch_result]).numpy()
                    batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                    if args.Embed == 1: validation_embeddings.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                    loss = criterion(pred, target)
                
                elif args.model_type == 'Hr':
                    ###### GET PREDICTION
                    if not args.molecular:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                else:
                                    pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs, PGgs,Hr)
                                else:
                                    pred,Rmap = predictor(RGgs, PGgs,Hr)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                else:
                                    pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs, PGgs,Hr)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                    else:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,Hr,RAdd)
                                else:
                                    pred,Rmap = predictor(RGgs,Hr,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,Hr)
                                else:
                                    pred,Rmap = predictor(RGgs,Hr)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,Hr,RAdd)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,Hr)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs,Hr)

                
                    # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                    batch_result = torch.cat([pred,target]).cpu().data
                    batch_result = torch.cat([smiles,batch_result]).numpy()
                    batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                    if args.Embed == 1: validation_embeddings.append(embeddings.cpu().numpy().tolist())
                    test.append(batch_result)
                    loss = criterion(pred, target)
                
                elif args.model_type == 'Hr_multi':
                    ###### GET PREDICTION
                    if not args.molecular:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                else:
                                    pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs, PGgs,Hr)
                                else:
                                    pred,Rmap = predictor(RGgs, PGgs,Hr)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                else:
                                    pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs, PGgs,Hr)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                    else:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,Hr,RAdd)
                                else:
                                    pred,Rmap = predictor(RGgs,Hr,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,Hr)
                                else:
                                    pred,Rmap = predictor(RGgs,Hr)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,Hr,RAdd)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,Hr)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs,Hr)

                    # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                    batch_result = torch.cat([pred,target]).cpu().data
                    batch_result = torch.cat([smiles,batch_result]).numpy()
                    batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                    if args.Embed == 1: validation_embeddings.append(embeddings.cpu().numpy().tolist())
                    
                    test.append(batch_result)
                    if len(args.tweights) == len(args.targets):
                        loss = None 
                        for index,weight in enumerate(args.tweights):
                            p = pred[:,index].unsqueeze(1)
                            t = target[:,index].unsqueeze(1)
                            if loss is not None:
                                loss += args.tweights[index] * criterion(p,t)
                            else:
                                loss = args.tweights[index] * criterion(p,t)

                    else:
                        print('Cannot work becauze the weights are underdetermined.')
                        return None 

                elif args.model_type == 'multi':
                    ###### GET PREDICTION
                    if not args.molecular:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs, PGgs)
                                else:
                                    pred,Rmap = predictor(RGgs, PGgs)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs, PGgs)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs, PGgs)

                    else:
                        if args.Embed == 0:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs,RAdd)
                                else:
                                    pred,Rmap = predictor(RGgs,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred = predictor(RGgs)
                                else:
                                    pred,Rmap = predictor(RGgs)
                        elif args.Embed == 1:
                            if args.additionals is not None:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs,RAdd)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs,RAdd)
                            else:
                                if not args.AttnMaps:
                                    pred,embeddings = predictor(RGgs)
                                else:
                                    pred,embeddings,Rmap = predictor(RGgs)
                    

                    # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                    batch_result = torch.cat([pred,target]).cpu().data
                    batch_result = torch.cat([smiles,batch_result]).numpy()
                    batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                    if args.Embed == 1: validation_embeddings.append(embeddings.cpu().numpy().tolist())
                    
                    test.append(batch_result)
                    if len(args.tweights) == len(args.targets):
                        loss = None
                        for index,weight in enumerate(args.tweights):
                            p = pred[:,index].unsqueeze(1)
                            t = target[:,index].unsqueeze(1)
                            if loss is not None:
                                loss += args.tweights[index] * criterion(p,t)
                            else:
                                loss = args.tweights[index] * criterion(p,t)
                    else:
                        print('Cannot work becauze the weights are underdetermined.')
                        return None 

                val_loss.append(loss.cpu().data.numpy())

            logger.info('Validation accuracy is: %.5f' % np.mean(val_loss))

            if not args.test_only: 
                if not args.molecular:
                    ext = []
                    columns = ['ID','RTYPE','Rsmiles','Psmiles','Rinchi','Pinchi']
                    if isinstance(args.target,list):
                        preds  = [t+'_PRED' for t in args.target]
                        columns += preds        
                        columns += args.target
                    else:
                        columns += [args.target+'_PRED']
                        columns += [args.target]
                else:
                    ext = []
                    columns = ['ID','RTYPE','Rsmiles','Rinchi']
                    if isinstance(args.target,list):
                        preds  = [t+'_PRED' for t in args.target]
                        columns += preds        
                        columns += args.target
                    else:
                        columns += [args.target+'_PRED']
                        columns += [args.target]
                   
                ext_loss = []
                ext_embeddings = []

                for item in tqdm(extDataLoader, total=len(extDataLoader), smoothing=0.9):
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
                                    id,rtypes,Rgs,smiles,targets,Radd = item
                            
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
                                    id,rtypes,Rgs,Pgs,smiles,targets = item
                                else:
                                    id,rtypes,Rgs,smiles,targets = item
                                     
                    if args.model_type == 'BEP':
                        ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                        target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
                        if isinstance(args.additionals,list):
                            logger.info('Error: BEP-like Prediciton can only be done on one additional set of values.')
                            break
                        else:
                            Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

                    elif args.model_type == 'direct':
                        ###### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                        target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
                    
                    elif args.model_type == 'Hr':
                        ##### TAKE THE FIRST COLUMN (USUALLY THE DE) AND LOAD IT TO CUDA
                        target    = torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)

                        ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                        if isinstance(args.additionals,list):
                            if args.Norm is not None:
                                Hr        = additionals.float().view(args.batch_size,len(args.additionals)).numpy()
                                Hr        = scaler.transform(Hr).to(device)

                            else:
                                Hr        = additionals.float().view(args.batch_size,len(args.additionals)).to(device)
                        else:
                            if args.Norm is not None:
                                Hr        = additionals.float().view(args.batch_size,1).numpy()
                                Hr        = scaler.transform(Hr).to(device)
                                
                            else:
                                Hr = torch.Tensor([float(i[0]) for i in additionals]).view(args.batch_size,1).to(device)

                    elif args.model_type == 'multi':
                        ###### LOAD ALL TARGETS TO CUDA
                        if isinstance(args.target,list):
                            target = targets.float().view(args.batch_size,len(args.target)).to(device)
                        else:
                            logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                            break 

                    elif args.model_type == 'Hr_multi':
                            ###### LOAD ALL TARGETS TO CUDA
                        if isinstance(args.target,list):
                            target = targets.float().view(args.batch_size,len(args.target)).to(device)
                        else:
                            logger.info('Error: Need at least two variables to run this. Otherwise it does not work')
                            break 
                        ##### TAKE THE ADDITIONAL FEATURES AND LOAD THEM INTO CUDA
                        if isinstance(args.additionals,list):
                            if args.Norm is not None:
                                Hr        = additionals.float().view(args.batch_size,len(args.additionals)).numpy()
                                Hr        = scaler.transform(Hr).to(device)

                            else:
                                Hr        = additionals.float().view(args.batch_size,len(args.additionals)).to(device)
                        else:
                            if args.Norm is not None:
                                Hr        = additionals.float().view(args.batch_size,1).numpy()
                                Hr        = scaler.transform(Hr).to(device)
                                
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
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,PGgs,RAdd,PAdd)
                                    else:
                                        pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs, PGgs)
                                    else:
                                        pred,Rmap = predictor(RGgs, PGgs)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                                    else:
                                        pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs, PGgs)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs, PGgs)

                        else:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,RAdd)
                                    else:
                                        pred,Rmap = predictor(RGgs,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs)
                                    else:
                                        pred,Rmap = predictor(RGgs)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,RAdd)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs)
                    
                            
                        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                        batch_result = torch.cat([pred,target]).cpu().data
                        batch_result = torch.cat([smiles,batch_result]).numpy()
                        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                        if args.Embed == 1: ext_embeddings.append(embeddings.cpu().numpy().tolist())
                        ext.append(batch_result)
                        loss = criterion(pred, target)
                    
                    elif args.model_type == 'BEP':
                        ###### GET PREDICTION
                        if not args.molecular:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        outputs = predictor(RGgs,PGgs,RAdd,PAdd)
                                    else:
                                        outputs,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                                else:
                                    if not args.AttnMaps:
                                        outputs = predictor(RGgs, PGgs)
                                    else:
                                        outputs,Rmap = predictor(RGgs, PGgs)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        outputs,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                                    else:
                                        outputs,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    if not args.AttnMaps:
                                        outputs,embeddings = predictor(RGgs, PGgs)
                                    else:
                                        outputs,embeddings,Rmap = predictor(RGgs, PGgs)

                        else:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        outputs = predictor(RGgs,RAdd)
                                    else:
                                        outputs,Rmap = predictor(RGgs,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        outputs = predictor(RGgs)
                                    else:
                                        outputs,Rmap = predictor(RGgs)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        outputs,embeddings = predictor(RGgs,RAdd)
                                    else:
                                        outputs,embeddings,Rmap = predictor(RGgs,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        outputs,embeddings = predictor(RGgs)
                                    else:
                                        outputs,embeddings,Rmap = predictor(RGgs)
                    
                        pred = outputs[:, 0].unsqueeze(1) * Hr + outputs[:, 1].unsqueeze(1)
                        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                        batch_result = torch.cat([pred,target]).cpu().data
                        batch_result = torch.cat([smiles,batch_result]).numpy()
                        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                        if args.Embed == 1: ext_embeddings.append(embeddings.cpu().numpy().tolist())
                        
                        ext.append(batch_result)
                        loss = criterion(pred, target)
                    
                    elif args.model_type == 'Hr':
                        ###### GET PREDICTION
                        if not args.molecular:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                    else:
                                        pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs, PGgs,Hr)
                                    else:
                                        pred,Rmap = predictor(RGgs, PGgs,Hr)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                    else:
                                        pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs, PGgs,Hr)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                        else:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,Hr,RAdd)
                                    else:
                                        pred,Rmap = predictor(RGgs,Hr,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,Hr)
                                    else:
                                        pred,Rmap = predictor(RGgs,Hr)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,Hr,RAdd)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,Hr)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs,Hr)

                        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                        batch_result = torch.cat([pred,target]).cpu().data
                        batch_result = torch.cat([smiles,batch_result]).numpy()
                        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                        if args.Embed == 1: ext_embeddings.append(embeddings.cpu().numpy().tolist())
                        
                        ext.append(batch_result)
                        loss = criterion(pred, target)
                    
                    elif args.model_type == 'Hr_multi':
                        ###### GET PREDICTION
                        if not args.molecular:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                    else:
                                        pred,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)

                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs, PGgs,Hr)
                                    else:
                                        pred,Rmap = predictor(RGgs, PGgs,Hr)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                    else:
                                        pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,Hr,RAdd,PAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs, PGgs,Hr)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs, PGgs,Hr)

                        else:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,Hr,RAdd)
                                    else:
                                        pred,Rmap = predictor(RGgs,Hr,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,Hr)
                                    else:
                                        pred,Rmap = predictor(RGgs,Hr)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,Hr,RAdd)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs,Hr,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,Hr)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs,Hr)

                        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                        batch_result = torch.cat([pred,target]).cpu().data
                        batch_result = torch.cat([smiles,batch_result]).numpy()
                        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                        if args.Embed == 1: ext_embeddings.append(embeddings.cpu().numpy().tolist())
                        
                        ext.append(batch_result)
                        if len(args.tweights) == len(args.targets):
                            loss = None
                            for index,weight in enumerate(args.tweights):
                                p = pred[:,index].unsqueeze(1)
                                t = target[:,index].unsqueeze(1)
                                if loss is not None:
                                    loss += args.tweights[index] * criterion(p,t)
                                else:
                                    loss = args.tweights[index] * criterion(p,t)
                        else:
                            print('Cannot work becauze the weights are underdetermined.')
                            return None 

                    elif args.model_type == 'multi':
                        ###### GET PREDICTION
                        if not args.molecular:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,PGgs,RAdd,PAdd)
                                    else:
                                        pred,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)

                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs, PGgs)
                                    else:
                                        pred,Rmap = predictor(RGgs, PGgs)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,PGgs,RAdd,PAdd)
                                    else:
                                        pred,embeddings,Rmap,Pmap = predictor(RGgs,PGgs,RAdd,PAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs, PGgs)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs, PGgs)

                        else:
                            if args.Embed == 0:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs,RAdd)
                                    else:
                                        pred,Rmap = predictor(RGgs,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred = predictor(RGgs)
                                    else:
                                        pred,Rmap = predictor(RGgs)
                            elif args.Embed == 1:
                                if args.additionals is not None:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs,RAdd)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs,RAdd)
                                else:
                                    if not args.AttnMaps:
                                        pred,embeddings = predictor(RGgs)
                                    else:
                                        pred,embeddings,Rmap = predictor(RGgs)
                    

                        # Merge the torch Tensors, get them out of cuda, and move them to numpy 
                        batch_result = torch.cat([pred,target]).cpu().data
                        batch_result = torch.cat([smiles,batch_result]).numpy()
                        batch_result = np.hstack((id,rtypes,batch_result)).tolist()
                        if args.Embed == 1: ext_embeddings.append(embeddings.cpu().numpy().tolist())
                        
                        ext.append(batch_result)
                        if len(args.tweights) == len(args.targets):
                            loss = None
                            for index,weight in enumerate(args.tweights):
                                p = pred[:,index].unsqueeze(1)
                                t = target[:,index].unsqueeze(1)
                                if loss is not None:
                                    loss += args.tweights[index] * criterion(p,t)
                                else:
                                    loss = args.tweights[index] * criterion(p,t)
                        else:
                            print('Cannot work becauze the weights are underdetermined.')
                            return None 

                    ext_loss.append(loss.cpu().data.numpy())
      
                logger.info('Test accuracy is: %.5f' % np.mean(ext_loss))
        
            # compute the average
        if args.weightsandbiases:
            if args.test_only:
                wandb.log({"learning_rate":lr, "train_loss": train_instance_acc, 'val_loss': np.mean(val_loss)},step=epoch)
            else:
                wandb.log({"learning_rate":lr, "train_loss": train_instance_acc, 'val_loss': np.mean(val_loss),'ext_loss': np.mean(ext_loss)},step=epoch)
        
        logger.info('Epoch %d Test MAE: %fkcal/mol' % (epoch + 1, np.mean(val_loss)))
        if not args.test_only: logger.info('Epoch %d External MAE: %fkcal/mol' % (epoch + 1, np.mean(ext_loss)))

        if np.mean(val_loss) < best_loss:
            loss_increase_count = 0
            best_loss = np.mean(val_loss)
            logger.info('Save model...')
            savepath = 'best_model.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': best_loss,
                'model_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')


            ###### SAVE RESULTS AS CSV
            traincsv = pd.DataFrame(train,columns=columns)
            traincsv.to_csv('best_train.csv')

            testcsv = pd.DataFrame(test,columns=columns)
            testcsv.to_csv('best_val.csv')

            if not args.test_only:
                extcsv = pd.DataFrame(ext,columns=columns)
                extcsv.to_csv('best_test.csv')

            
            ##### SAVE OVERALL RESULTS AS SCORES. 
            training_embeddings = pd.DataFrame(training_embeddings)
            training_embeddings.to_csv('train_embeddings.csv')

            validation_embeddings = pd.DataFrame(validation_embeddings)
            training_embeddings.to_csv('validation_embeddings.csv')
            
            if not args.test_only:
                ext_embeddings = pd.DataFrame(ext_embeddings)
                ext_embeddings.to_csv('test_embeddings.csv')
            
        if loss_increase_count > args.patience:
            break

        logger.info('Best MAE is: %.5f' % best_loss)
        global_epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to the config file")

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
    main(config)
