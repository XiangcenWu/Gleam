import os
from TrainInference import data_spilt, ReadH5Pkld, get_loader
from TrainInference import train_net, data_all, train_net_contrast, lora_finetune, get_roc
from monai.transforms import *

from torch.optim import AdamW

import torch
import torch.nn as nn
import logging
import argparse


from vit_classifier import vit_classifier, apply_lora

import torch.nn.functional as F
from Losses import FocalLoss, SimpleContrastiveLoss

import random

parser = argparse.ArgumentParser(description="Choose a network for training.")
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",  # Default to 'cuda:0'
    help="Choose the device (e.g., 'cuda:0', 'cuda:1', 'cpu')"
)
parser.add_argument(
    "--mode",
    type=str,
    default="lora",  # Default to 'cuda:0'
    help="Choose the device (e.g., 'lora', 'full', 'none', 'full-no-contrast')"
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=16,  # Default batch size
    help="Set batch size for training"
)
parser.add_argument(
    "--seed",
    type=int,
    default=325,  # Default to 'swin'
    help=f"Choose a seed to generate dataset"
)


# Parse arguments
args = parser.parse_args()
device = args.device
FTmode = args.mode
batchsize = args.batchsize
seed = args.seed


log_name = f'log_{FTmode}_{seed}'
logging.basicConfig(
    filename=os.path.join('logs', log_name+".txt"),  # Log file
    filemode="w",  # Overwrite if exists
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)



train_list, inference_list = data_spilt('/raid/candi/xiangcen/patient_level/promise_patient_level', 462, seed)


train_transform = Compose([
        ReadH5Pkld(),
        RandAffined(['img'], spatial_size=(128, 128, 64), prob=0.5, shear_range=(0., 0.5, 0., 0.5, 0., 0.2), mode='nearest', padding_mode='zeros'),
        RandGaussianSmoothd(['img'], prob=0.25),
        RandGaussianNoised(['img'], prob=0.25, std=0.05),
        RandAdjustContrastd(['img'], prob=0.25, gamma=(0.5, 2.))
    ])

inference_transform = ReadH5Pkld()



train_loader = get_loader(train_list, train_transform, batch_size=batchsize)
inference_loader = get_loader(inference_list, inference_transform, batch_size=1, shuffle=False, drop_last=False)

contract_loss_function = SimpleContrastiveLoss(margin=0.2)

model = vit_classifier(6)
# whether to load the base model
if FTmode in ['full', 'full-no-contrast']:
    model.load_state_dict(torch.load('/home/xiangcen/PatientBiopsyDetect/models/model_weights_pirAgle_68.pth', map_location=device))

if FTmode == 'lora':
    for param in model.parameters():
        param.requires_grad = False
    model = apply_lora(model, rank=4)
    
# put the right parameters to optimizer 
if FTmode == 'lora':
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
elif FTmode in ['full' , 'none', 'full-no-contrast']:
    optimizer = AdamW(model.parameters(), lr=1e-4)

# do contrast learning or not
if FTmode == 'full-no-contrast':
    with_contrast=False
else:
    with_contrast=True
    
    
    
loss_function  = FocalLoss(gamma=2)
for e in range(11):
    loss = lora_finetune(
            model=model,
            train_loader=train_loader,
            train_optimizer=optimizer,
            train_loss=loss_function,
            contrast_loss=contract_loss_function,
            with_contrast=with_contrast,
            device=device,
        )


    tpr, tnr, auroc = get_roc(
        model,
        inference_loader,
        device=device,
        steps = 19
    )
    logging.info(f'epoch {e}, loss{loss}, auc {auroc}')
    
    
    torch.save(model.state_dict(), f"./models/model_{FTmode}_{seed}.pth")

