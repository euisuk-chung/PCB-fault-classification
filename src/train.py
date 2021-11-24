import os
import csv
import cv2
import numpy as np
from time import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # automatic mixed precision (to accelerate computation)

from torchvision import transforms
from torchvision.models import resnet50
from warmup_scheduler import GradualWarmupScheduler

from utils.imageprocess import image_transformer, image_processor
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader
from utils.radams import RAdam
from tqdm import tqdm
import IPython


class CustomSmoothedLoss:
    def __init__(self):
        self.base_loss_function = nn.MultiLabelSoftMarginLoss()
        
    def __call__(self, pred, gt, eps=0.1):
        
        smoothed_gt = torch.where(gt==0, eps, 1-eps)
        
        loss1 = self.base_loss_function(pred, smoothed_gt)
        
        sum_preds = torch.sum(pred > 0.5, axis=-1)
        sum_gt = torch.sum(gt, axis=-1)
        loss2 = torch.mean(torch.abs(sum_gt - sum_preds))
        
        fin_loss = loss1 + 0.1*loss2
        
        return fin_loss

    
class CustomLoss:
    def __init__(self):
        self.base_loss_function = nn.MultiLabelSoftMarginLoss()
        
    def __call__(self, pred, gt):
        # basic loss function
        loss1 = self.base_loss_function(pred, gt)

        sum_preds = torch.sum(pred > 0.5, axis=-1)
        sum_gt = torch.sum(gt, axis=-1)
        # 예측 개수 penalty
        loss2 = torch.mean(torch.abs(sum_gt - sum_preds))
        
        # gt가 1인데 0으로 예측하는 경우에 penalty 크게 (gt-pred==1 인 경우)
        pred = torch.where(pred < 0.5, 0 ,1)
        loss3 = torch.mean(torch.sum(gt-pred, axis=-1))

        return loss1 + 0.1*loss2 + 0.3*loss3
    

def train_model(input_model, fold_k, model_save_path, args, logger, *loaders):
    
    fold_k = fold_k+1
    
    model = input_model
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    logger = logger
    train_loader = loaders[0]
    val_loader = loaders[1]
    
    #early_stopping = EarlyStopping(patience=args.patience, verbose=False, fold_k=fold_k, path=model_save_path)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    
    # -----------------
    #   amp wrapping
    # -----------------
    scaler = amp.GradScaler()

    # LERANING RATE SCHEDULER (WARMUP)
    decay_rate = 0.97

    if args.lr_type == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
    
    elif args.lr_type == 'cos':
        annealing_cycle = 3
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=int(args.epochs/annealing_cycle))
    
    elif args.lr_type == 'multi':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=[
                                                                int(args.epochs*0.3),
                                                                int(args.epochs*0.4),
                                                                int(args.epochs*0.6)
                                                            ],
                                                            gamma=0.7)

        
        
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    
    logger.info(f"""
                    ---------------------------------------------------------------------------
                        TRAINING INFO
                            Loss function : {loss_function}
                            Optimizer     : {optimizer}
                            LR_Scheduler  : {lr_scheduler}
                    ---------------------------------------------------------------------------""")
    
    train_tot_num = train_loader.dataset.__len__()
    val_tot_num = val_loader.dataset.__len__()

    train_corrects_sum = 0
    train_loss_sum = 0.0
    
    val_corrects_sum = 0
    val_loss_sum = 0.0
    
    logger.info(f'Training begins... Epochs = {epochs}')
    
    for epoch in range(epochs):

        time_start = time()
        
        # warmup scheduler step / lr scheduler
        ### BASELINE은 없게
        #if epoch <= warmup_epochs:
        #    lr_warmup.step()
        
        logger.info(f"""
===========================================================================
    PHASE INFO
        Current fold  : Fold ({fold_k})
        Current phase : {epoch+1}th epoch
        Learning Rate : {optimizer.param_groups[0]['lr']:.6f}
---------------------------------------------------------------------------""")
        
        torch.cuda.empty_cache()
        model.train()
        
        train_tmp_num = 0
        train_tmp_corrects_sum = 0
        train_tmp_loss_sum = 0.0
        for idx, (train_X, train_Y) in enumerate(train_loader):

            train_tmp_num += len(train_Y)
            
            optimizer.zero_grad()
            with amp.autocast():
                train_pred = model(train_X)
                train_loss = loss_function(train_pred, train_Y)
                
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_pred_label = train_pred > args.threshold
            train_corrects = (train_pred_label == train_Y).sum()
            train_corrects_sum += train_corrects.item()
            
            train_loss_sum += train_loss.item()
            
            train_tmp_corrects_sum += train_corrects.item()
            train_tmp_loss_sum += train_loss.item()
            
            # Check between batches
            verbose = args.verbose
            if (idx+1) % verbose == 0:
                print(f"-- ({str((idx+1)).zfill(4)} / {str(len(train_loader)).zfill(4)}) Train Loss: {train_tmp_loss_sum/(idx+1):.6f} | Train Acc: {train_tmp_corrects_sum/(train_tmp_num*6)*100:.4f}%")
                
        # initialization
        train_tmp_num = 0
        train_tmp_corrects_sum = 0
        train_tmp_loss_sum = 0.0

        with torch.no_grad():
            
            for idx, (val_X, val_Y) in enumerate(val_loader):
                
                with amp.autocast():
                    val_pred = model(val_X)
                    val_loss = loss_function(val_pred, val_Y)

                val_pred_label = val_pred > args.threshold
                val_corrects = (val_pred_label == val_Y).sum()
                val_corrects_sum += val_corrects.item()
                val_loss_sum += val_loss.item()
                
        train_acc = train_corrects_sum/(train_tot_num*6)*100
        train_loss = train_loss_sum/len(train_loader)
        
        val_acc = val_corrects_sum/(val_tot_num*6)*100
        val_loss = val_loss_sum/len(val_loader)
        
        time_end = time()
        time_len_m, time_len_s = divmod(time_end - time_start, 60)
        
        logger.info(f"""
---------------------------------------------------------------------------
    SUMMARY
        Finished phase  : {epoch+1}th epoch
        Time taken      : {time_len_m:.0f}m {time_len_s:.2f}s
        Training Loss   : {train_loss:.6f}  |  Training Acc   : {train_acc:.4f}%
        Validation Loss : {val_loss:.6f}  |  Validation Acc : {val_acc:.4f}%
===========================================================================\n""")
        
        if ((epoch+1) >= epochs-10) and ((epoch+1) % 2 == 0):
            save_path = os.path.join(model_save_path, f'model_ckpt_fold{fold_k}_{epoch+1}.pth')
            logger.info(f"SAVING MODEL: {save_path}")
            torch.save(model.state_dict(), save_path)
        
#         # EARLY STOPPER
#         early_stopping(val_loss, model)
#         if early_stopping.early_stop:
#             logger.info("Early stopping condition met --- TRAINING STOPPED")
#             logger.info(f"Best score: {early_stopping.best_score}")
#             break
        
        # INITIALIZATION
        train_corrects_sum = 0
        val_corrects_sum = 0
        
        train_loss_sum = 0.0
        val_loss_sum = 0.0
  
        # Save training result
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        
        # lr scheduler step
        lr_scheduler.step()

    return model, train_loss_list, val_loss_list, train_acc_list, val_acc_list