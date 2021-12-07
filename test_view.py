from mrn                             import mrn
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

def test(BASE_PATH, BATCH_SIZE, CLASSES, WEIGHT_PATH):
    ### Using GPU Device
    GPU = True
    device = "cuda" if GPU and torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    ### Model Setting 
    model = mrn(in_channels=3, class_num=CLASSES)
    # model = torch.nn.DataParallel(model, device_ids=[0,1]) 
    model.cuda()

    # Path Setting
    Path = PathSplit(BASE_PATH)
    TRAIN_ZIP, VALID_ZIP, TEST_ZIP = Path.Split()

    # Dataset, DataLoader Customizing
    test_dataset = PathToDataset(TEST_ZIP, (512,512), augmentation=None)               
    test_x, test_y, test_ph = test_dataset.NumpyDataset()
    test_data = TensorData(test_x, test_y, test_ph)


    ### DataLoader
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

    ### Test best saved model
    best_model = torch.load(WEIGHT_PATH)
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=device,
    )
    logs = test_epoch.run(test_loader)
    print(logs)    

    return TEST_ZIP, device, best_model

if __name__ == '__main__':
    ### Argparse Setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_PATH', type=str, required=True, help='Input the patch path. It should be like ../slideset/patientnum/mask/patches.png.')
    parser.add_argument('--BATCH_SIZE', default=8, type=int, required=False, help='Input the batch size.')
    parser.add_argument('--CLASSES', default=3, type=int, required=True, help='Input the class of the patches. It should be 1 or >2.')
    parser.add_argument('--WEIGHT_PATH', default='./', help='Input the Weight path.')
    args = parser.parse_args()

    ### Argparse to Variable
    BASE_PATH = args.BASE_PATH
    BATCH_SIZE = args.BATCH_SIZE
    CLASSES = args.CLASSES
    WEIGHT_PATH = args.WEIGHT_PATH

    ### Let's test
    test(BASE_PATH, BATCH_SIZE, CLASSES, WEIGHT_PATH)