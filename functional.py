from datetime                     import date
import matplotlib.pyplot          as plt
import numpy                      as np
import torch
import cv2

def name_weight(frame=None, classes=None, description=None):
    mode = 'binary_class' if classes==1 else 'multi_class'
    if frame == 'tensorflow':
        try:
            os.makedirs('../../model/tensorflow')
            return '../../model/tensorflow/' + f'{mode}_mrn_{description}_{date.today()}.h5'
        except:
            return '../../model/tensorflow/' + f'{mode}_mrn_{description}_{date.today()}.h5'
    elif frame == 'pytorch':
        try:
            os.makedirs('../../model/pytorch')
            return '../../model/pytorch/' + f'{mode}_mrn_{description}_{date.today()}.pth'
        except:
            return '../../model/pytorch/' + f'{mode}_mrn_{description}_{date.today()}.pth'
    else:
        raise NameError('Check frame or model. frame should be tensorflow or pytorch. model should be model`s name.')

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, vmin=0, vmax=2, cmap='Oranges')
    plt.show()

def predict(TEST_ZIP, device, best_model):
    for i in range(len(TEST_ZIP)):
        
        image_vis = cv2.imread(TEST_ZIP[i][0]).astype('uint8')

        image_batch = np.zeros((2,) + (512, 512) + (3,), dtype='float32')
        image_batch[0,...] = cv2.imread(TEST_ZIP[i][0])
        image_batch[1,...] = cv2.imread(TEST_ZIP[i][1])
        gt_mask = cv2.imread(TEST_ZIP[i][2], 0)
        image_batch = torch.from_numpy(image_batch).float().to(device).unsqueeze(0).permute(0,1,4,2,3)

        pr_mask = best_model(image_batch)
        pr_mask = pr_mask.squeeze().permute(1,2,0).cpu().detach().numpy().round()
        pr_mask = np.argmax(pr_mask, axis=2)

        visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )


def multi_predict(**model):
    TEST_ZIP = model['TEST_ZIP']
    MODEL = model['MODEL']
    device = model['device']
    del model['TEST_ZIP']
    del model['MODEL']
    del model['device']
    ### setting
    ENCODER = 'se_resnext101_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax'
    n = len(model) + 2
    for i in range(len(TEST_ZIP)): 
        model_dict = dict()
        image_vis = cv2.imread(TEST_ZIP[i][0]).astype('uint8')

        image_batch1 = np.zeros((512, 512) + (3,), dtype='float32')
        image_batch1 = cv2.imread(TEST_ZIP[i][0])
        image_batch1 = torch.from_numpy(image_batch1).float().to(device).unsqueeze(0).permute(0,3,1,2)

        image_batch2 = np.zeros((2,) + (512, 512) + (3,), dtype='float32')
        image_batch2[0,...] = cv2.imread(TEST_ZIP[i][0])
        image_batch2[1,...] = cv2.imread(TEST_ZIP[i][2])
        image_batch2 = torch.from_numpy(image_batch2).float().to(device).unsqueeze(0).permute(0,1,4,2,3)

        image_batch4 = np.zeros((4,) + (512, 512) + (3,), dtype='float32')
        image_batch4[0,...] = cv2.imread(TEST_ZIP[i][0])
        image_batch4[1,...] = cv2.imread(TEST_ZIP[i][1])
        image_batch4[2,...] = cv2.imread(TEST_ZIP[i][2])
        image_batch4[3,...] = cv2.imread(TEST_ZIP[i][3])
        image_batch4 = torch.from_numpy(image_batch4).float().to(device).unsqueeze(0).permute(0,1,4,2,3)

        gt_mask = cv2.imread(TEST_ZIP[i][4], 0)          
        model_dict['image'] = image_vis
        model_dict['ground_truth_mask'] = gt_mask

        for name, model_i in model.items():
            # if name == 'Unet' or 'UnetPlusPlus' or 'DeepLabV3Plus' or 'MANet':
            #     model_dict[name] = model_i
            #     try:
            #         model_dict[name] = model_dict[name].module.predict(image_batch1)
            #     except:
            #         model_dict[name] = model_dict[name].predict(image_batch1)
            #     model_dict[name] = (model_dict[name].squeeze().cpu().numpy().round())
            #     model_dict[name] = np.argmax(model_dict[name], axis=0)
            # elif name == 'hooknet' or 'hooknet_seresnext101':
            #     model_dict[name] = model_i
            #     model_dict[name] = model_dict[name](image_batch2)
            #     model_dict[name] = model_dict[name].squeeze().permute(1,2,0).cpu().detach().numpy().round()
            #     model_dict[name] = np.argmax(model_dict[name], axis=2)
            # elif name == 'quad_scale_hooknet':
            model_dict[name] = model_i
            try:
                try:
                    model_dict[name] = model_dict[name].module.predict(image_batch1)
                except:
                    model_dict[name] = model_dict[name].predict(image_batch1)
                model_dict[name] = (model_dict[name].squeeze().cpu().numpy().round())
                model_dict[name] = np.argmax(model_dict[name], axis=0)
            except:
                try:
                    model_dict[name] = model_i
                    model_dict[name] = model_dict[name](image_batch2)
                    model_dict[name] = model_dict[name].squeeze().permute(1,2,0).cpu().detach().numpy().round()
                    model_dict[name] = np.argmax(model_dict[name], axis=2)
                except:
                    model_dict[name] = model_i
                    model_dict[name] = model_dict[name](image_batch4)
                    model_dict[name] = model_dict[name].squeeze().permute(1,2,0).cpu().detach().numpy().round()
                    model_dict[name] = np.argmax(model_dict[name], axis=2)

        plt.figure(figsize=(16,5))
        for i, (name, image) in enumerate(model_dict.items()):
            plt.subplot(1, n, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image, vmin=0, vmax=2, cmap='Oranges')
        plt.show()
