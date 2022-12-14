import torch
import glob

MODEL         = 'MRN'
ENCODER_NAME  = 'se_resnext101_32x4d'
CLASSES       = 6
BASE_PATH     = glob.glob('/data/AGGC_2022/multi_tissue30_tumor05/Subset3_Train_9_Zeiss/input_y100/*.png')
INPUT_SHAPE   = 512
SAMPLER       = None
BATCH_SIZE    = 16
NUM_WORKER    = 4
LOSS          = 'TverskyLoss'
DESCRIPTION   = 'mrn_model'
LR            = 1e-4
OPTIMIZER     = 'Adam'
EPOCH         = 100
INDICE        = 1
MAGNIFICATION = 100
GPU           = True
DEVICE        = "cuda" if GPU and torch.cuda.is_available() else "cpu"