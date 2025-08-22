import os
from TrainInference import data_spilt, ReadH5Pkld, get_loader
from TrainInference import train_net, data_all, train_net_contrast, get_roc
from monai.transforms import *

from torch.optim import AdamW

import torch
import torch.nn as nn
import logging
import argparse


from vit_classifier import vit_classifier

import torch.nn.functional as F
from Losses import FocalLoss, SimpleContrastiveLoss

import random
from torch.cuda.amp import autocast

parser = argparse.ArgumentParser(description="Choose a network for training.")
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",  # Default to 'cuda:0'
    help="Choose the device (e.g., 'cuda:0', 'cuda:1', 'cpu')"
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
batchsize = args.batchsize
seed = args.seed


log_name = 'log_basepirAgle'
logging.basicConfig(
    filename=os.path.join('logs', log_name+".txt"),  # Log file
    filemode="w",  # Overwrite if exists
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


# train_list, inference_list = data_spilt('/raid/candi/xiangcen/data_all_modality_update_crop_three', 725, seed=seed)
train_list = data_all('/raid/candi/xiangcen/patient_level/paicai_patient_level', 
                     '/raid/candi/xiangcen/patient_level/ucla_patient_level',
                     '/raid/candi/xiangcen/patient_level/miama_patient_level')

_, inference_list = data_spilt('/raid/candi/xiangcen/patient_level/promise_patient_level', 462, seed)

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



model = vit_classifier(6)
# model.load_state_dict(torch.load('/home/xiangcen/PatientBiopsyDetect/models/model_weights_base_pirAgle.pth', map_location=device))
loss_function  = FocalLoss(gamma=2)
optimizer = AdamW(model.parameters(), lr=1e-4)



contract_loss_function = SimpleContrastiveLoss(margin=0.2)

for e in range(100):
    loss = train_net(
            model=model,
            train_loader=train_loader,
            train_optimizer=optimizer,
            train_loss=loss_function,
            contrast_loss = contract_loss_function,
            device=device,
        )



    tpr, tnr, auroc = get_roc(
        model,
        inference_loader,
        device=device,
        steps = 19
    )
    logging.info(f'epoch {e}, loss{loss}, auc {auroc}')
    
    torch.save(model.state_dict(), "./models/model_weights_pirAgle.pth")

