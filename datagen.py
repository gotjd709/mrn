from torch.utils.data                         import Dataset
from sklearn.utils                            import shuffle
from albumentations                           import *
import numpy                                  as np
import torch
import glob
import cv2

class PathSplit(object):
    def __init__(self, base_path=None):
        self.BASE_PATH = base_path

    def Split(self):
        TRAIN_ZIP = shuffle([(x, 
                      '/'.join(x.split('/')[:-2]) + '/input_x2/' + x.split('/')[-1], 
                      '/'.join(x.split('/')[:-2]) + '/input_y1/' + x.split('/')[-1]) for x in sorted(glob.glob(self.BASE_PATH  + '/input_x1/*.png'))], random_state=321)[:int(0.60*len(glob.glob(self.BASE_PATH  + '/input_x1/*.png')))]
        VALID_ZIP = shuffle([(x, 
                      '/'.join(x.split('/')[:-2]) + '/input_x2/' + x.split('/')[-1], 
                      '/'.join(x.split('/')[:-2]) + '/input_y1/' + x.split('/')[-1]) for x in sorted(glob.glob(self.BASE_PATH  + '/input_x1/*.png'))], random_state=321)[int(0.60*len(glob.glob(self.BASE_PATH  + '/input_x1/*.png'))):int(0.80*len(glob.glob(self.BASE_PATH  + '/input_x1/*.png')))]
        TEST_ZIP = shuffle([(x, 
                      '/'.join(x.split('/')[:-2]) + '/input_x2/' + x.split('/')[-1], 
                      '/'.join(x.split('/')[:-2]) + '/input_y1/' + x.split('/')[-1]) for x in sorted(glob.glob(self.BASE_PATH  + '/input_x1/*.png'))], random_state=321)[int(0.80*len(glob.glob(self.BASE_PATH  + '/input_x1/*.png'))):]
        return TRAIN_ZIP, VALID_ZIP, TEST_ZIP

class PathToDataset(object):
    def __init__(self, path_list, image_size, augmentation=None):
        self.path_list = path_list
        self.image_size = image_size
        self.augmentation = train_aug() if augmentation else test_aug()

    def NumpyDataset(self):
        batch_x = np.zeros((len(self.path_list),) + (2,) + self.image_size + (3,), dtype='float32')
        batch_y = np.zeros((len(self.path_list),) + self.image_size + (3,), dtype='float32')
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
            sample1 = self.augmentation(image=batch_x[j][0], mask=batch_y[j])
            sample2 = self.augmentation(image=batch_x[j][1])
            batch_x[j][0], batch_x[j][1], batch_y[j] = sample1['image'], sample2['image'], sample1['mask']
        print(f'batch_x.shape :: {batch_x.shape}, batch_y.shape :: {batch_y.shape}')
        return batch_x, batch_y, self.path_list

class TensorData(Dataset):    
    def __init__(self, x_data, y_data, path_list):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(0,1,4,2,3)
        self.y_data = torch.FloatTensor(y_data)
        self.y_data = self.y_data.permute(0,3,1,2)
        self.path_list = path_list
        self.len = self.y_data.shape[0]
        
    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

def train_aug():
    ret = Compose([
        ElasticTransform(alpha=1, sigma=20, alpha_affine=15, interpolation=1, border_mode=1),
        GridDistortion(),
        HorizontalFlip(),
        OpticalDistortion(),
        ShiftScaleRotate(),
        VerticalFlip()
    ])
    return ret

def test_aug():
    ret = Compose([
    ])
    return ret