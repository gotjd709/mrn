"""
train.py

Main smp(segmentation_models.pytorch) training for gastric cancer patch image and mask (512X512 10X) script.

Usage:
    train.py []
    train.py (-h | --help)
    train.py --version

Options:
    -h --help   Show this string.
    --version   Show version

"""
from mrn                             import mrn, mrn_seresnext101
from datagen                         import PathSplit, PathToDataset, TensorData
from functional                      import name_weight
from torchsampler                    import ImbalancedDatasetSampler
from torch.utils.data                import DataLoader
import segmentation_models_pytorch   as smp
import matplotlib.pyplot             as plt
import pandas                        as pd
import numpy                         as np
import argparse
import torch
import tqdm
import glob
import os
import cv2

### Argparse Setting
parser = argparse.ArgumentParser()
parser.add_argument('--BASE_PATH', type=str, required=True, help='Input the patch path. It should be like ../slideset/patientnum/mask/patches.png.')
parser.add_argument('--BACKBONE', type=str, required=True, help='Select the backbone model of MRN')
parser.add_argument('--BATCH_SIZE', default=4, type=int, required=False, help='Input the batch size.')
parser.add_argument('--CLASSES', default=3, type=int, required=True, help='Input the class of the patches. It should be 1 or >2.')
parser.add_argument('--EPOCHS', default=40, type=int, required=False, help='Input the epoches.')
parser.add_argument('--DESCRIPTION', type=str, required=True, help='Input the description of your trials briefly.')
args = parser.parse_args()

### Argparse to Variable
BASE_PATH = args.BASE_PATH
BACKBONE = args.BACKBONE
BATCH_SIZE = args.BATCH_SIZE
CLASSES = args.CLASSES
EPOCHS = args.EPOCHS
DESCRIPTION = args.DESCRIPTION

### Using GPU Device
GPU = True
device = "cuda" if GPU and torch.cuda.is_available() else "cpu"
print(f'Using device {device}')

### Model Setting 
if BACKBONE == 'vgg16':
    model = mrn(in_channels=3, class_num=CLASSES)
    # model = torch.nn.DataParallel(model, device_ids=[0,1]) 
    model.cuda()
elif BACKBONE == 'seresnext101':
    model = mrn_seresnext101(class_num=CLASSES, pretrained=None)
    # model = torch.nn.DataParallel(model, device_ids=[0,1]) 
    model.cuda()

# Path Setting
Path = PathSplit(BASE_PATH)
TRAIN_ZIP, VALID_ZIP, TEST_ZIP = Path.Split()

# Dataset, DataLoader Customizing
train_dataset = PathToDataset(TRAIN_ZIP, (512,512), augmentation=True)
valid_dataset = PathToDataset(VALID_ZIP, (512,512), augmentation=None)
test_dataset = PathToDataset(TEST_ZIP, (512,512), augmentation=None)
train_x, train_y, train_ph = train_dataset.NumpyDataset()
train_data = TensorData(train_x, train_y, train_ph)
valid_x, valid_y, valid_ph = valid_dataset.NumpyDataset()
valid_data = TensorData(valid_x, valid_y, valid_ph)                  
test_x, test_y, test_ph = test_dataset.NumpyDataset()
test_data = TensorData(test_x, test_y, test_ph)

### DataLoader
train_loader = torch.utils.data.DataLoader(
    train_data,
    sampler=ImbalancedDatasetSampler(train_data),
    batch_size=BATCH_SIZE,
    shuffle=False
)
valid_loader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

### Loss, Metrics, Optimizer Setting
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(), smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy(), smp.utils.metrics.Recall(), smp.utils.metrics.Precision(),
]
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True,
)
valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=device,
    verbose=True,
)
weight = name_weight(frame='pytorch', classes=CLASSES, description=DESCRIPTION)
max_score = 0
for i in range(0, EPOCHS):    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)   
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, weight)
        print('Model saved!')        
    if i == 50:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

### Test best saved model
best_model = torch.load(weight)
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=device,
)
logs = test_epoch.run(test_loader)    
