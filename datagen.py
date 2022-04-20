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

class PathToDataset(object):
    def __init__(self, path_list, image_size):
        self.path_list = path_list
        self.image_size = image_size

    def NumpyDataset(self):
        batch_x = np.zeros((len(self.path_list),) + (2,) + self.image_size + (3,), dtype='float32')
        batch_y = np.zeros((len(self.path_list),) + self.image_size + (4,), dtype='float32')
        for j, path in enumerate(self.path_list):
            img_path1 = path[0]
            img_path2 = path[1]
            mask_path = path[2]
            mask = cv2.imread(mask_path, 0)
            batch_x[j][0] = cv2.imread(img_path1)
            batch_x[j][1] = cv2.imread(img_path2)
            batch_y[j][:,:,0] = np.where(mask==0, 1, 0)
            batch_y[j][:,:,1] = np.where(mask==1, 1, 0)
            batch_y[j][:,:,2] = np.where(mask==2, 1, 0)
            batch_y[j][:,:,3] = np.where(mask==3, 1, 0)
        return batch_x, batch_y, self.path_list


class TensorData(Dataset):    
    def __init__(self, x_data, y_data, path_list, image_size, augmentation=None):
        self.x_data = x_data
        self.y_data = y_data
        self.path_list = path_list
        self.image_size = image_size
        self.len = self.y_data.shape[0]
        self.augmentation = train_aug() if augmentation else test_aug()
        
    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        sample = self.augmentation(image=self.x_data[index][0], image1=self.x_data[index][1], mask=self.y_data[index])
        x_data_s = np.zeros((2,) + self.image_size + (3,), dtype='float32')
        x_data_s[0], x_data_s[1], y_data_s = sample['image'], sample['image1'], sample['mask']
        x_data_s = torch.FloatTensor(x_data_s)
        x_data_s = x_data_s.permute(0,3,1,2)
        y_data_s = torch.FloatTensor(y_data_s)
        y_data_s = y_data_s.permute(2,0,1)
        return x_data_s, y_data_s

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