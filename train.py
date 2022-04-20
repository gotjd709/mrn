"""
train.py
Main smp(segmentation_models.pytorch) training for gastric cancer patch image and mask (512X512 10X) script.
"""
from model.mrn                       import mrn
from model.mrn_se_resnext101_32x4d   import mrn_se_resnext101_32x4d
from datagen                         import PathSplit, PathToDataset, TensorData
from functional                      import name_weight
from torchsampler                    import ImbalancedDatasetSampler
from torch.utils.data                import DataLoader
from torch.utils.tensorboard         import SummaryWriter
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

def pixel_ratio(BASE_PATH):
    zero_sum = 0; one_sum = 0; two_sum = 0; three_sum = 0   

    for patch in glob.glob(BASE_PATH):
        patch_image = cv2.imread(patch, 0)
        unique, counts = np.unique(patch_image, return_counts=True)
        uniq_cnt_dict = dict(zip(unique, counts))
        if 0 in uniq_cnt_dict:
            zero_sum += uniq_cnt_dict[0]
        if 1 in uniq_cnt_dict:
            one_sum += uniq_cnt_dict[1]
        if 2 in uniq_cnt_dict:
            two_sum += uniq_cnt_dict[2]
        if 3 in uniq_cnt_dict:
            three_sum += uniq_cnt_dict[3]
    return zero_sum, one_sum, two_sum, three_sum

def train(BASE_PATH, BACKBONE, BATCH_SIZE, CLASSES, MULTIPLE, EPOCHS, LOSS_FUNCTION, DESCRIPTION):
    ### Using GPU Device
    GPU = True
    device = "cuda" if GPU and torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    ### Model Setting 
    if BACKBONE == 'vgg16':
        model = mrn(in_channels=3, class_num=CLASSES, multiple=MULTIPLE)
        # model = torch.nn.DataParallel(model, device_ids=[0,1]) 
        model.cuda()
    elif BACKBONE == 'se_resnext101_32x4d':
        model = mrn_se_resnext101_32x4d(class_num=CLASSES)
        # model = torch.nn.DataParallel(model, device_ids=[0,1]) 
        model.cuda()
    else: 
        raise NameError('Please select the backbone within vgg16 or se_resnext101_32x4d')

    # Path Setting
    Path = PathSplit(BASE_PATH, MULTIPLE)
    TRAIN_ZIP, VALID_ZIP, TEST_ZIP = Path.Split()

    # Dataset, DataLoader Customizing
    train_dataset = PathToDataset(TRAIN_ZIP, (512,512))
    valid_dataset = PathToDataset(VALID_ZIP, (512,512))
    test_dataset = PathToDataset(TEST_ZIP, (512,512))
    train_x, train_y, train_ph = train_dataset.NumpyDataset()
    train_data = TensorData(train_x, train_y, train_ph, (512,512), augmentation=True)
    valid_x, valid_y, valid_ph = valid_dataset.NumpyDataset()
    valid_data = TensorData(valid_x, valid_y, valid_ph, (512,512), augmentation=None)                  
    test_x, test_y, test_ph = test_dataset.NumpyDataset()
    test_data = TensorData(test_x, test_y, test_ph, (512,512), augmentation=None)

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

    # Loss, Metrics, Optimizer Setting
    zero_sum, one_sum, two_sum, three_sum = pixel_ratio(BASE_PATH)
    weights = torch.tensor([zero_sum, one_sum, two_sum, three_sum], dtype=torch.float32)
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()
    if LOSS_FUNCTION == 'celoss':
        loss = torch.nn.CrossEntropyLoss(weight=weights)
        loss.__name__ = 'reweighted_celoss'
    elif LOSS_FUNCTION == 'diceloss':
        loss = smp.utils.losses.DiceLoss()
    else:
        raise NameError('Please select the loss function within celoss or diceloss')
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
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    ### Using Tensorboard
    writer = SummaryWriter(log_dir='../../log/MRN_se', filename_suffix=DESCRIPTION)
    writer.add_graph(model, images.cuda())

    max_score = 0
    for i in range(0, EPOCHS):    
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        if LOSS_FUNCTION == 'celoss':
            writer.add_scalars('Loss', {'train_loss':train_logs['reweighted_celoss'],
                                    'valid_loss':valid_logs['reweighted_celoss']}, i)
        else:
            writer.add_scalars('Loss', {'train_loss':train_logs['dice_loss'],
                            'valid_loss':valid_logs['dice_loss']}, i)
        writer.add_scalars('IoU', {'train_loss':train_logs['iou_score'],
                                    'valid_loss':valid_logs['iou_score']}, i)
        writer.add_scalars('Fscore', {'train_loss':train_logs['fscore'],
                                    'valid_loss':valid_logs['fscore']}, i)
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, weight)
            print('Model saved!')        
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    ### Summary writer closing
    writer.close()

    ### Test best saved model
    best_model = torch.load(weight)
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=device,
    )
    logs = test_epoch.run(test_loader) 

if __name__ == '__main__':
    ### Argparse Setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_PATH', type=str, required=True, help='Input the patch path. It should be like ../slideset/patientnum/mask/patches.png.')
    parser.add_argument('--BACKBONE', type=str, required=True, help='Select the backbone model of MRN')
    parser.add_argument('--BATCH_SIZE', default=4, type=int, required=False, help='Input the batch size.')
    parser.add_argument('--CLASSES', default=3, type=int, required=True, help='Input the class of the patches. It should be 1 or >2.')
    parser.add_argument('--MULTIPLE', default=1, type=int, required=False, help='Input the value of (context_mpp)/(target_mpp*2).')
    parser.add_argument('--EPOCHS', default=50, type=int, required=False, help='Input the epoches.')
    parser.add_argument('--LOSS_FUNCTION', default='celoss', help='Input the loss function for model. It should be celoss or diceloss.')
    parser.add_argument('--DESCRIPTION', type=str, required=True, help='Input the description of your trials briefly.')
    args = parser.parse_args()

    ### Argparse to Variable
    BASE_PATH = args.BASE_PATH
    BACKBONE = args.BACKBONE
    BATCH_SIZE = args.BATCH_SIZE
    CLASSES = args.CLASSES
    MULTIPLE = args.MULTIPLE
    EPOCHS = args.EPOCHS
    LOSS_FUNCTION = args.LOSS_FUNCTION
    DESCRIPTION = args.DESCRIPTION

    train(BASE_PATH, BACKBONE, BATCH_SIZE, CLASSES, MULTIPLE, EPOCHS, LOSS_FUNCTION, DESCRIPTION)