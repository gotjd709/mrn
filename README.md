# MRN (Multi-Resolution Network)

This is **my personal implementation of MRN**. I also summarize [MRN paper](https://arxiv.org/pdf/1807.09607.pdf) in [my blog](https://biology-statistics-programming.tistory.com/162).

### Structure
> Two novel multi-resolution networks are proposed to learn from
input patches extracted at multiple levels. These patches share the same centroid
and shape (size in pixels), but with an octave based increase of the pixel size,
micrometers per pixel (mpp). Only the central high resolution patch is segmented
at the output. - [Feng Gu et al. (2018)](https://arxiv.org/pdf/1807.09607.pdf)

<p align="center"><img src="https://user-images.githubusercontent.com/70703320/147843606-7e371fad-dd9c-4cc7-b34e-007efcc62c63.png" width="60%" height="30%"></p>

### Environment
```
pip install -r requirements.txt
```
Above, I install python 3.6 with CUDA 11.4

# Description

### Repository Structure
- `model/mrn.py`: main MRN model script
- `model/mrn_se_resnext101_32x4d.py`: main MRN model script
- `datagen.py`: data dataloader and augmentation script
- `functional.py`: sub train function script 
- `train.py`: main train script

### Training

##### Data Preparation
```
MRN_Data
    ├ slide_num_1
    |       ├ input_x100
    |       ├ input_x50
    |       └ input_y100
    .
    .
    .
    └ slide_num_n
            ├ input_x100
            ├ input_x50
            └ input_y100   
```
- input_x1: mpp=1 image patches(512x512) directory
- input_x2: mpp=2 image patches(512x512) directory
- input_y1: mpp=1 mask patches(512x512) directory 

</br>

You can get this data structure by using [util_multi.py](https://github.com/CODiPAI-CMC/wsi_processing)

##### Train Example
```
python train.py 
```
You can adjust hyper parameters in config.py

# Reference

### paper
- [Model Structure](https://arxiv.org/pdf/1807.09607.pdf)

### code
- [SE-ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)
- [Training](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb)