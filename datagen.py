from torch.utils.data                         import Dataset
from sklearn.utils                            import shuffle
from albumentations                           import *
import numpy                                  as np
import torch
import glob
import cv2

class PathSplit(object):
    def __init__(self, base_path=None, multiple=1):
        self.BASE_PATH = glob.glob(base_path)
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
        if self.multiple == 1:
            TRAIN_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x2/'+x.split('/')[-1], x) for x in TRAIN_TARGET_PATH],
                random_state=333
            )
            VALID_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x2/'+x.split('/')[-1], x) for x in VALID_TARGET_PATH],
                random_state=333
            )
            TEST_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x2/'+x.split('/')[-1], x) for x in TEST_TARGET_PATH],
                random_state=333
            )
        elif self.multiple == 2:
            TRAIN_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x3/'+x.split('/')[-1], x) for x in TRAIN_TARGET_PATH],
                random_state=333
            )
            VALID_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x3/'+x.split('/')[-1], x) for x in VALID_TARGET_PATH],
                random_state=333
            )
            TEST_ZIP = shuffle(
            [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x3/'+x.split('/')[-1], x) for x in TEST_TARGET_PATH],
                random_state=333
            )
        return TRAIN_ZIP, VALID_ZIP, TEST_ZIP


class TensorData(Dataset):    
    def __init__(self, path_list, image_size, augmentation=None):
        self.path_list = path_list
        self.image_size = image_size
        self.augmentation = train_aug() if augmentation else test_aug()
        
    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, index):
        batch_x = np.zeros((1,) + (2,) + self.image_size + (3,), dtype='float32')
        batch_y = np.zeros((1,) + self.image_size + (4,), dtype='float32')
        img_path1, img_path2, mask_path = self.path_list[index]
        batch_x[0][0] = cv2.imread(img_path1)
        batch_x[0][1] = cv2.imread(img_path2)
        mask = cv2.imread(mask_path, 0)
        batch_y[0][:,:,0] = np.where(mask==0, 1, 0)
        batch_y[0][:,:,1] = np.where(mask==1, 1, 0)
        batch_y[0][:,:,2] = np.where(mask==2, 1, 0)
        batch_y[0][:,:,3] = np.where(mask==3, 1, 0)      
        sample = self.augmentation(image=batch_x[0][0], image1=batch_x[0][1], mask=batch_y[0])
        data_x = np.zeros((2,) + self.image_size + (3,), dtype='float32')
        data_x[0], data_x[1], data_y = sample['image'], sample['image1'], sample['mask']
        data_x = torch.FloatTensor(data_x)
        data_x = data_x.permute(0,3,1,2)
        data_y = torch.FloatTensor(data_y)
        data_y = data_y.permute(2,0,1)
        return data_x, data_y

def train_aug():
    ret = Compose(
        [
            OneOf([
                HorizontalFlip(p=1),
                OpticalDistortion(p=1),
                VerticalFlip(p=1)
            ], p=0.75)
        ],
        additional_targets={'image1':'image'}
    )
    return ret

def test_aug():
    ret = Compose([
    ])
    return ret