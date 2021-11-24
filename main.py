import os
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
import random
from time import time
import IPython
import copy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from warmup_scheduler import GradualWarmupScheduler

from src.train import train_model
from utils.imageprocess import image_transformer, image_processor
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import CustomDataLoader
from utils.radams import RAdam
from utils.call_model import CallModel
from utils.visualize import plot_result

#from utils.pretrained_model import CustomModel, CallPretrainedModel

from tqdm import tqdm
import logging

# for oversampling train dataset
from torchsampler import ImbalancedDatasetSampler

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

# train/val splitting method
def split_index(total_index, val_ratio):
    tot_len = len(total_index)
    train_len = int(tot_len*(1-val_ratio))

    train_sampled = random.sample(total_index, train_len)
    val_sampled = [i for i in total_index if i not in train_sampled]
    logger.info(f"Trainset length: {len(train_sampled)}, Valset length: {len(val_sampled)}")
    return train_sampled, val_sampled

# split kfold
def split_kfold(k, train_len=2000):    
    kfold = KFold(n_splits=k, shuffle=True)
    splitted = kfold.split(range(train_len))
    return splitted

# load dataset
def load_trainset(train_df=None, mode='train', device=None, train_index=None, val_index=None, batch_size=16):
    '''
    mode: train or train_ovr
    '''
    if mode in ['train', 'train_ovr']:
        # train w/o oversampling
        train_set = CustomDataLoader(train_df, train=True, row_index=train_index, device=device)
        val_set = CustomDataLoader(train_df, train=False, row_index=val_index, device=device)
        
        if mode == 'train':
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        elif mode == 'train_ovr':
            train_loader = torch.utils.data.DataLoader(train_set,
                                                       sampler=ImbalancedDatasetSampler(train_set),
                                                       batch_size=batch_size)
            
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
        return train_loader, val_loader

    
def load_testset(test_df=None, test_index=None, tta=False, angles=[], device=None):
    if not tta:
        tmp_test_set = CustomDataLoader(test_df, train=False, row_index=test_index, device=device)
        tmp_test_loader = torch.utils.data.DataLoader(tmp_test_set, batch_size=16, shuffle=False)
        test_loader = [tmp_test_loader]
    else:
        test_loader = []
        for angle in angles:
            tmp_test_set = CustomDataLoader(test_df, row_index=test_index, device=device, tta=True, angle=angle)
            tmp_test_loader = torch.utils.data.DataLoader(tmp_test_set, batch_size=16, shuffle=False)
            test_loader.append(tmp_test_loader)

    return test_loader


def load_trained_weight(model_input=None, model_index=0, fold_k=1, model_type='early', trained_weight_path='./ckpt'):
    assert model_index > 0

    model_name = f'early_stopped_fold{fold_k}.pth' if model_type == 'early' else f'model_ckpt_fold{fold_k}_{model_type}.pth'
    ckpt_path = os.path.join(trained_weight_path, f'model_{model_index}', model_name)
    print(ckpt_path)

    trained_model = model_input
    trained_model.load_state_dict(torch.load(ckpt_path))
    trained_model.eval()

    return trained_model
    

def make_inference(args, model, test_loader, test_tot_num):
    
    total_set = []

    test_corrects_sum = 0
    test_loss_sum = 0.0

    for idx, single_test_loader in enumerate(test_loader):
        logger.info(f"Inference on test_loader ({idx+1}/{len(test_loader)})")

        fin_labels = []
        for _, (test_X, test_Y) in enumerate(single_test_loader):
            
            # Make predictions
            with torch.no_grad():
                pred = model(test_X)

            if args.voting == 'soft':
                pred_label = torch.sigmoid(pred).detach().to('cpu').numpy()
            else:
                pred_label = ((pred > args.threshold)*1).detach().to('cpu').numpy()
        
            pred_label = pred > args.threshold
            test_corrects = (pred_label == test_Y).sum()
            test_corrects_sum += test_corrects.item()

            pred_label = pred_label.detach().to('cpu').numpy()
            fin_labels.append(pred_label)
            torch.cuda.empty_cache()
        
        test_acc = test_corrects_sum/(test_tot_num*6)*100
        
        logger.info(f"""
---------------------------------------------------------------------------
    SUMMARY
        Test Acc        : {test_acc:.4f}%
===========================================================================\n""")        
        logger.info("Done.")

        test_corrects_sum = 0
        test_loss_sum = 0.0

        total_set.append(np.concatenate(fin_labels))
    
    if args.voting == 'soft':
        fin_total_set = np.mean(total_set, axis=0)
    else:
        fin_total_set = np.where(np.mean(total_set, axis=0) >= 0.5, 1, 0)
    
    return fin_total_set
 
    
def aggregate_submit(args, predictions):
    # Aggregation(voting)
    agg = np.where(np.mean(predictions, axis=0) >= 0.5, 1, 0)
    submission_file = pd.DataFrame(agg)
    save_path = os.path.join(args.base_dir, f'submit/submission_model_{args.model_index}.csv')

    submission_file.to_csv(save_path, index=False)
    logger.info(f'Result file save at: {save_path}')



if __name__ == "__main__":
    
    # ARGUMENTS PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_index", type=int, default=0, help='My model index. Integer type, and should be greater than 0')
    parser.add_argument("--base_dir", type=str, default="/repo/course/sem21_01/PCB-fault-classification", help='Base PATH of your work')
    parser.add_argument("--label_dir", type=str, default="/repo/course/sem21_01/PCB-fault-classification/label.csv", help='label PATH')
    parser.add_argument("--mode", type=str, default="train", help='[train | train_ovr | test]')
    parser.add_argument("--data_type", type=str, default="original", help='[original | denoised]: default=denoised')
    parser.add_argument("--ckpt_path", type=str, default="/repo/course/sem21_01/PCB-fault-classification/ckpt", help='PATH to weights of ckpts.')
    parser.add_argument("--base_model", type=str, default="plain_resnet50", help="[plain_resnet50, custom_resnet50, plain_efficientnetb4]")
    parser.add_argument("--pretrained", dest='pretrained', action='store_true', help='Default is false, so specify this argument to use pretrained model')
    parser.add_argument("--pretrained_weights_dir", type=str, default="/home/ys/repo/PCB-fault-classification/pretrained_model", help='PATH to weights of pretrained model')
    parser.add_argument("--cuda", dest='cuda', action='store_false', help='Whether to use CUDA: defuault is True, so specify this argument not to use CUDA')
    parser.add_argument("--device_index", type=int, default=0, help='Cuda device to use. Used for multiple gpu environment')
    parser.add_argument("--batch_size", type=int, default=8, help='Batch size for train-loader for training phase')
    parser.add_argument("--test_ratio", type=float, default=0.20, help='Ratio for testset: default=0.20')
    parser.add_argument("--val_ratio", type=float, default=0.15, help='Ratio for validation set: default=0.15')
    parser.add_argument("--epochs", type=int, default=10, help='Epochs for training: default=100')
    parser.add_argument("--learning_rate", type=float, default=0.003, help='Learning rate for training: default=0.0029')
    parser.add_argument("--lr_type", choices= ['exp','cos','multi'], help='Type of learing rate scheduler')
    parser.add_argument("--patience", type=int, default=10, help='Patience of the earlystopper: default=10')
    parser.add_argument("--verbose", type=int, default=100, help='Between batch range to print train accuracy: default=100')
    parser.add_argument("--threshold", type=float, default=0.0, help='Threshold used for predicting 0/1')
    parser.add_argument("--seed", type=int, default=227182, help='Seed used for reproduction')
    parser.add_argument("--fold_k", type=int, default=1, help='Number of fold for k-fold split. If k=1, standard train/val splitting is done.')
    parser.add_argument("--tta", dest='tta', action='store_true', help='Whether to use TTA on inference. Specify this argument to use TTA.')
    parser.add_argument("--voting", type=str, default='hard', help='Choosing soft voting or hard voting at inference')
    args = parser.parse_args()
    
    
    # ASSERT CONDITIONS
    # added train_ovr
    # assert (args.model_index > 0) and (args.mode in ['train', 'train_ovr', 'test'])
    
    
    LOG_PATH = os.path.join(args.base_dir, 'logs')
    
    #IPython.embed();exit(1);
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s : %(message)s", 
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(LOG_PATH, f"log_model_{args.model_index}.txt")),
                            logging.StreamHandler()
                        ])
    logger.info("START")
    
    # ------------------------
    #   GLOBAL CUDA SETTING
    # ------------------------
    global_cuda = args.cuda and torch.cuda.is_available()
    global_device = torch.device(f'cuda:{args.device_index}') if global_cuda else torch.device('cpu')

    logger.info(f"Global Device: {global_device}")
    logger.info(f'Parsed Args: {args}')
    
    
    # ------------------
    #    Seed Setting
    # ------------------
    set_seed(args.seed)
    
    
    # -----------------------
    #      SET DIRECTORY
    # -----------------------
    base_dir = args.base_dir
    ckpt_folder_path = os.path.join(args.ckpt_path, f'model_{args.model_index}')

    try:
        os.mkdir(ckpt_folder_path)
    except FileExistsError:
        print(f'{ckpt_folder_path} alreay exist!')
        pass
    
    # -------------------
    #   TRAIN/VAL SPLIT
    # -------------------
    label_df = pd.read_csv(args.label_dir)
    tot_num = label_df.shape[0]
    
    train_num = int(tot_num*(1-args.test_ratio))
    test_num = tot_num - train_num
    logger.info(f"Trainset(train+val) length: {train_num}, Testset length: {test_num}")
    
    train_sample_idx = random.sample(range(tot_num), train_num)
    test_sample_idx = [idx for idx in range(tot_num) if idx not in train_sample_idx]
    
    train_df = label_df.iloc[train_sample_idx]
    test_df = label_df.iloc[test_sample_idx]
    test_index = range(len(test_df))

    if args.fold_k == 1:
        train_index_set, val_index_set = split_index(range(len(train_df)), args.val_ratio)
        train_index_set, val_index_set = [train_index_set], [val_index_set]
        
    elif args.fold_k > 1:
        splitted = split_kfold(args.fold_k, train_len=len(train_df))
        train_index_set, val_index_set = [], []
        
        for train_fold, val_fold in splitted:
            train_index_set.append(train_fold)
            val_index_set.append(val_fold)
        
        logger.info(f"Trainset length: {len(train_index_set[0])}, Valset length: {len(val_index_set[0])}")


    # ----------------
    #    Call model
    # ----------------
    base_model_type = args.base_model
    base_model = CallModel(model_type=base_model_type,
                           pretrained=args.pretrained,
                           logger=logger,
                           path=args.pretrained_weights_dir).model_return()
    
    model = base_model.to(global_device)
    
    
    # --------------------
    #        TRAIN
    # --------------------
    if args.mode in ['train', 'train_ovr']:
        
        #  MAKE FOLDER for saving CHECKPOINTS
        # if folder already exists, assert. Else, make folder.
        # assert not os.path.exists(ckpt_folder_path), "Model checkpoint folder already exists."
        # os.makedirs(ckpt_folder_path)
        for k in range(args.fold_k):
            model_to_train = copy.deepcopy(model)
            
            logger.info(f"Training on Fold ({k+1}/{args.fold_k})")
            # Load trainset/valset
            train_index = train_index_set[k]
            val_index = val_index_set[k]
            
            train_loader, val_loader = load_trainset(train_df=train_df,
                                                     mode=args.mode,
                                                     batch_size=args.batch_size,
                                                     train_index=train_index,
                                                     val_index=val_index,
                                                     device=global_device)
            
            # Train model
            _, train_loss_list, val_loss_list, train_acc_list, val_acc_list = train_model(model_to_train,
                                                                                          k,
                                                                                          ckpt_folder_path,
                                                                                          args,
                                                                                          logger,
                                                                                          train_loader,
                                                                                          val_loader)
            
            # Plot
            plot_result(args.model_index, k+1, train_loss_list, train_acc_list, val_loss_list, val_acc_list)

            
    # --------------------
    #      INFERENCE
    # -------------------    
    if args.mode == 'test':
        test_loader = load_testset(test_df=test_df,
                                   test_index=test_index,
                                   tta = args.tta,
                                   angles = [0, 90, -90, 180],
                                   device=global_device)
        
        pred_list = []
        for k in range(args.fold_k):
            logger.info(f"Inference using model of fold ({k+1}/{args.fold_k})")
            
            # Call model & trained weights
            model_inference = load_trained_weight(model_input=model,
                                                  model_index=args.model_index,
                                                  model_type=50,
                                                  fold_k=k+1
                                                ).to(global_device)
        
            pred = make_inference(args, model_inference, test_loader, test_num)
            pred_list.append(pred)
        
        # Aggregate and Save result
        aggregate_submit(args, pred_list)