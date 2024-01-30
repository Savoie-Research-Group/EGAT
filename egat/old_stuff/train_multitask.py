"""
Author: Qiyuan Zhao
Date: Feb 2023
"""
import argparse
import os
import torch
import logging
import sys
import importlib
import shutil
import numpy as np

import wandb

from tqdm import tqdm
from dataset import RGD1Dataset,collate
import hydra
import omegaconf

from torch.autograd import Variable

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@hydra.main(config_path='config', config_name='EGAT_multi_3MLP')
def main(args):
    
    omegaconf.OmegaConf.set_struct(args, False)
    wandb.init(project='EGAT_milti')
    torch.set_grad_enabled(True) 
    print(torch.cuda.is_available())
    device = torch.device("cuda")
    logger = logging.getLogger(__name__)

    # attemp to find a restart point
    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        logger.info('Use pretrain model, starting from epoch {}'.format(start_epoch))
        best_loss = checkpoint['test_acc']

    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logger.info('No existing model, starting training from scratch...')
        best_loss = 100
        start_epoch = 0

    '''HYPER PARAMETER'''
    root = hydra.utils.to_absolute_path('/depot/bsavoie/data/Qiyuan/TS_pred/GAT/RGD1-VB')

    # load in exlucde items
    exclude = []
    with open('/depot/bsavoie/data/Qiyuan/TS_pred/GAT/RGD1-VB/exclude.txt','r') as f:
        for lc,lines in enumerate(f):
            exclude.append(lines.split('/')[-1].split('.json')[0])

    #TRAIN_DATASET = RGD1Dataset(root=root, split='train', class_choice='R1P1')
    TRAIN_DATASET = RGD1Dataset(root=root, split='train', class_choice=None, return_RxnE=True, exclude=exclude)
    # drop_last: drop the last incomplete batch, if the dataset size is not divisible by the batch size
    # shuffle: have the data reshuffled at every epoch
    # num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True, collate_fn=collate)
    #TEST_DATASET = RGD1Dataset(root=root, split='val', class_choice='R1P1')
    TEST_DATASET = RGD1Dataset(root=root, split='val', class_choice=None, return_RxnE=True, exclude=exclude)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=True, collate_fn=collate)

    '''MODEL LOADING'''
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    print(torch.cuda.is_available())
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
        pass

    criterion = torch.nn.L1Loss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(predictor.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = args.learning_rate_min
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = args.lr_decay
    MOMENTUM_DECAY_STEP = args.step_size

    global_epoch = 0
    loss_increase_count = 0

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
        loss_list = []
        act_loss  = []
        for Rgs, Pgs, targets in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            
            target_act= torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
            target_hr = torch.Tensor([float(i[3]) for i in targets]).view(args.batch_size,1).to(device)
            RGgs      = Rgs.to(device)
            PGgs      = Pgs.to(device)
            RGgs.ndata['x'].to(device)
            RGgs.edata['x'].to(device)
            PGgs.ndata['x'].to(device)
            PGgs.edata['x'].to(device)
            optimizer.zero_grad()

            pred = predictor(RGgs, PGgs)
            pred_act, pred_hr = pred[:,0].unsqueeze(1), pred[:,1].unsqueeze(1)
            loss = criterion(pred_act, target_act) + 0.25 * criterion(pred_hr, target_hr)

            loss.backward()
            loss_list.append(loss.cpu().data.numpy())
            act_loss.append(criterion(pred_act, target_act).cpu().data.numpy())
            optimizer.step()

        train_instance_acc = np.mean(loss_list)
        train_act_acc      = np.mean(act_loss)
        logger.info('Train accuracy is: %.5f' % train_instance_acc)

        # model evaluation
        with torch.no_grad():

            test_metrics = {}
            predictor = predictor.eval()
            eval_loss = []
            hr_loss   = []
            for Rgs, Pgs, targets in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):

                target_act= torch.Tensor([float(i[2]) for i in targets]).view(args.batch_size,1).to(device)
                target_hr = torch.Tensor([float(i[3]) for i in targets]).view(args.batch_size,1).to(device)
                RGgs      = Rgs.to(device)
                PGgs      = Pgs.to(device)
                RGgs.ndata['x'].to(device)
                RGgs.edata['x'].to(device)
                PGgs.ndata['x'].to(device)
                PGgs.edata['x'].to(device)

                pred   = predictor(RGgs, PGgs)
                pred_act, pred_hr = pred[:,0].unsqueeze(1), pred[:,1].unsqueeze(1)
                eval_loss.append(criterion(pred_act, target_act).cpu().data.numpy())
                hr_loss.append(criterion(pred_hr, target_hr).cpu().data.numpy())

        # compute the average
        wandb.log({"learning_rate":lr, "train_loss": train_instance_acc, 'train_DE_MAE':train_act_acc, 'val_DE_MAE': np.mean(eval_loss), 'val_Hr_MAE':np.mean(hr_loss)},step=epoch)
        logger.info('Epoch %d test MAE: %fkcal/mol' % (epoch + 1, np.mean(eval_loss)))
        
        if np.mean(eval_loss) < best_loss:
            loss_increase_count = 0
            best_loss = np.mean(eval_loss)
            logger.info('Save model...')
            savepath = 'best_model.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'train_DE_acc': train_act_acc,
                'test_acc': best_loss,
                'model_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        if loss_increase_count > 10:
            break

        logger.info('Best MAE is: %.5f' % best_loss)
        global_epoch += 1

if __name__ == '__main__':
    main()
