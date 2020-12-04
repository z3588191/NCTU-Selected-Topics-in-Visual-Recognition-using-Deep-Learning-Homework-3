# NCTU Selected Topics in Visual Recognition using Deep Learning, Homework 3
Code for Instance Segmentation on Tiny PASCAL VOC dataset.


## Hardware
The following specs were used to create the submited solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- NVIDIA GeForce 2080Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#Installation)
2. [Dataset Download](#Dataset-Download)
3. [Prepapre Dataset](#Prepare-Dataset)
4. [Train models](#Train-models)
5. [Pretrained models](#Pretrained-models)
6. [Reference](#Reference)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hw3 python=3.7
source activate hw3
pip install -r requirements.txt
```

## Dataset Download
Dataset download link is in Data section of [HW3](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK)

## Prepare Dataset
After downloading, the data directory is structured as:
```
${ROOT}
  +- test_images
  |  +- 2007_000629.png
  |  +- 2007_001175.png
  |  +- ...
  +- train_images
  |  +- 2007_000033.png
  |  +- 2007_000042.png
  |  +- ...
  +- pascal_train.json
  +- test.json
```

### Train models
To train models, run following commands.
```
$ python train.py 
```


## Pretrained models
You can download pretrained model that used for my submission from [link](https://drive.google.com/drive/folders/1srX4rt_JvmTdIjEBcyAJ2CiMg_v2Jw-G?usp=sharing).
And put it in the directory :
```
${ROOT}
  +- ResNestWithFPN.pth
  +- test.py
  +- ...
```

To get prediction of testing data, run following commands.
Then you will get `submission.json`
```
$ python test.py 
```

## Reference
[torchvision](https://github.com/pytorch/vision)
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
[mmdetection](https://github.com/open-mmlab/mmdetection)
[ResNest](https://arxiv.org/abs/2004.08955)