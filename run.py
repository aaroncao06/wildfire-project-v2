import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import zarr
from pathlib import Path
import torch.optim as optim
# Import Project Modules -----------------------------------------------------------------------------------------------
from models import RAdam
#from Models.loss import get_loss_module
from dataloader import temporal_dataloader
from models import TemporalConvTran
from train import train



logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()

# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
args = parser.parse_args()

# config needs 'Data_shape', 'emb_size', 'num_heads', 'dim_ff'

if __name__ == '__main__':
    config = args.__dict__  # configuration dictionary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ------------------------------------ Load Data ---------------------------------------------------------------
    logger.info("Loading Data ...")
    
    dataset = xr.open_dataset(Path.home() / 'Documents/Wildfire Project/california_dataset.zarr', engine='zarr') #xarray
    total_timesteps = dataset.sizes['time']
    train_test_ratio = 0.75
    ratio_index = int(3 * total_timesteps // 4)
    train_dataset = dataset.isel(time=slice(None, ratio_index))
    test_dataset = dataset.isel(time=slice(ratio_index, None))
    feature_names = ['tp', 'rel_hum', 'ws10', 't2m_mean', 't2m_min', 't2m_max', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'lsm',
                     'drought_code_max', 'drought_code_mean', 'fwi_max', 'fwi_mean', 'lst_day', 'lai', 'ndvi', 'pop_dens',
                     'lccs_class_0', 'lccs_class_1', 'lccs_class_2', 'lccs_class_3', 'lccs_class_4', 'lccs_class_5', 'lccs_class_6', 'lccs_class_7', 'lccs_class_8']
    target_name = ['fcci_ba']
    lat_size = 4
    lon_size = 4
    time_size = 322 # a third of total time
    sequence_length = 64 #look up to 512 days back
    
    #def temporal_dataloader(dataset, feature_names, target_name, lat_size, lon_size, time_size, sequence_length, shuffle=True, num_workers=0):
    training_dataloader = temporal_dataloader(train_dataset, feature_names, target_name, lat_size, lon_size, time_size, sequence_length, shuffle=True, num_workers=0)
    test_dataloader = temporal_dataloader(test_dataset, feature_names, target_name, lat_size, lon_size, time_size, sequence_length, shuffle=True, num_workers=0)
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------

    logger.info("Creating model ...")
    config['Data_shape'] = [16, len(feature_names), sequence_length]
    config['num_labels'] = 1
    # config for model needs 'Data_shape', 'emb_size', 'num_heads', 'dim_ff', all have default vals except for data_shape
    model = TemporalConvTran(config, num_classes=config['num_labels'])
    #logger.info("Model:\n{}".format(model))
    #logger.info("Total number of parameters: {}".format(count_parameters(model)))
    # -------------------------------------------- Model Initialization ------------------------------------
    optimizer = RAdam(model.parameters(), lr=config['lr'], weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    # ---------------------------------------------- Training The Model ------------------------------------
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, training_dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.2f}%')
