from torch.utils.data                         import Dataset
from sklearn.utils                            import shuffle
from albumentations                           import Compose, OneOf, HorizontalFlip, VerticalFlip
from torchsampler                             import ImbalancedDatasetSampler
from config                                   import *
import numpy                                  as np
import torch
import cv2

class PathSplit(object):
    def __init__(self, base_path=None, multiple=1):
        self.BASE_PATH = base_path
        self.multiple = multiple

    def Split(self):
        # split path
        TRAIN_TARGET_PATH = shuffle(self.BASE_PATH, random_state=321)[:int(0.60*len(self.BASE_PATH))]
        VALID_TARGET_PATH = shuffle(self.BASE_PATH, random_state=321)[int(0.60*len(self.BASE_PATH)):int(0.80*len(self.BASE_PATH))]
        TEST_TARGET_PATH = shuffle(self.BASE_PATH, random_state=321)[int(0.80*len(self.BASE_PATH)):]      
        # check train, valid, test length
        print('TRAIN TOTAL :', len(TRAIN_TARGET_PATH))
        print('VALID TOTAL :', len(VALID_TARGET_PATH))
        print('TEST TOTAL :', len(TEST_TARGET_PATH))
        print('/'.join(TRAIN_TARGET_PATH[0].split('/')[:-2])+'/input_x100/'+TRAIN_TARGET_PATH[0].split('/')[-1])
        if self.multiple == 1:
            TRAIN_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x100/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x50/'+x.split('/')[-1], x) for x in TRAIN_TARGET_PATH],
                random_state=333
            )
            VALID_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x100/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x50/'+x.split('/')[-1], x) for x in VALID_TARGET_PATH],
                random_state=333
            )
            TEST_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x100/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x50/'+x.split('/')[-1], x) for x in TEST_TARGET_PATH],
                random_state=333
            )
        elif self.multiple == 2:
            TRAIN_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x100/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x25/'+x.split('/')[-1], x) for x in TRAIN_TARGET_PATH],
                random_state=333
            )
            VALID_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x100/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x25/'+x.split('/')[-1], x) for x in VALID_TARGET_PATH],
                random_state=333
            )
            TEST_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x100/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x25/'+x.split('/')[-1], x) for x in TEST_TARGET_PATH],
                random_state=333
            )
        return TRAIN_ZIP, VALID_ZIP, TEST_ZIP


class TensorData(Dataset):    
    def __init__(self, path_list, image_size, classes, augmentation=None):
        self.path_list = path_list
        self.image_size = (image_size,image_size)
        self.classes = classes
        self.augmentation = train_aug() if augmentation else test_aug()
        
    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, index):
        batch_x = np.zeros((2,) + self.image_size + (3,), dtype='float32') 
        x_data = batch_x.copy()
        batch_y = np.zeros(self.image_size + (self.classes,), dtype='uint8')
        img_path1, img_path2, mask_path = self.path_list[index]
        batch_x[0] = cv2.imread(img_path1)/255
        batch_x[1] = cv2.imread(img_path2)/255
        mask = cv2.imread(mask_path, 0)
        for i in range(self.classes):
            batch_y[...,i] = np.where(mask==i, 1, 0)
        sample = self.augmentation(image=batch_x[0], image1=batch_x[1], mask=batch_y) 
        x_data[0], x_data[1], y_data = sample['image'], sample['image1'], sample['mask']
        x_data = torch.FloatTensor(x_data)
        x_data = x_data.permute(0,3,1,2)
        y_data = torch.FloatTensor(y_data)
        y_data = y_data.permute(2,0,1)
        return x_data, y_data

def train_aug():
    ret = Compose(
        [
            OneOf([
                HorizontalFlip(p=1),
                VerticalFlip(p=1)
            ], p=0.66)
        ],
        additional_targets={'image1':'image'}
    )
    return ret

def test_aug():
    ret = Compose([
    ])
    return ret

def dataloader_setting():
    # Path Setting
    path = PathSplit(BASE_PATH)
    train_zip, valid_zip, test_zip = path.Split()
    train_data = TensorData(train_zip, INPUT_SHAPE, CLASSES, augmentation=True)
    valid_data = TensorData(valid_zip, INPUT_SHAPE, CLASSES)                  
    test_data = TensorData(test_zip, INPUT_SHAPE, CLASSES)
    ### DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler = ImbalancedDatasetSampler(train_data) if SAMPLER else None,
        batch_size=BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    return train_loader, valid_loader, test_loader