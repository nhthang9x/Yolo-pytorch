# [PYTORCH] YOLO (You Only Look Once)

## Introduction

Here is my pytorch implementation of the model described in the paper **You Only Look Once:
Unified, Real-Time Object Detection** [paper](https://arxiv.org/abs/1506.02640). 

## How to use my code
With my code, you can:
* **Train your model from scratch**
* **Train your model with my trained model**
* **Evaluate test images with either my trained model or yours**

## Datasets:

I used 4 different datases: VOC2007, VOC2012, COCO2014 and COCO2017. Statistics of datasets I used for experiments is shown below

| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| VOC2007                |    20   |      5011/12608       |           4952/-           |
| VOC2012                |    20   |      5717/13609       |           5823/13841       |
| COCO2014               |    80   |         83k/-         |            41k/-           |
| COCO2017               |    80   |         118k/-        |             5k/-           |

Create a data folder under the repository,

```
cd {repo_root}
mkdir data
```

- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:
  ```
  VOCDevkit
  ├── VOC2007
  │   ├── Annotations  
  │   ├── ImageSets
  │   ├── JPEGImages
  │   ├── ...
  └── VOC2012
      ├── Annotations  
      ├── ImageSets
      ├── JPEGImages
      └── ...
  ```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:
  ```
  COCO
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   └── instances_val2017.json
  │── images
  │   ├── train2014
  │   ├── train2017
  │   ├── val2014
  │   └── val2017
  └── anno_pickle
      ├── COCO_train2014.pkl
      ├── COCO_val2014.pkl
      ├── COCO_train2017.pkl
      └── COCO_val2017.pkl
  ```
  
## Setting:

* **Model structure**: In compared to the paper, I changed structure of top layers, to make it converge better. You could see the detail of my YoloNet in **src/yolo_net.py**.
* **Data augmentation**: I performed dataset augmentation, to make sure that you could re-trained my model with small dataset (~500 images). Techniques applied here includes HSV adjustment, crop, resize and flip with random probabilities
* **Loss**: The losses for object and non-objects are combined into a single loss in my implementation
* **Optimizer**: I used SGD optimizer and my learning rate schedule is as follows: 

|         Epoches        | Learning rate |
|------------------------|:---------------:|
|          0-4           |      1e-5     |
|          5-79          |      1e-4     |
|          80-109        |      1e-5     |
|          110-end       |      1e-6     |

## Training

For each dataset, I provide 2 different pre-trained models, which I trained with corresresponding dataset:
- **whole_model_trained_yolo_xxx**: The whole trained model.
- **only_params_trained_yolo_xxx**: The trained parameters only.

You could specify which trained model file you want to use, by the parameter **pre_trained_model_type**. The parameter **pre_trained_model_path** then is the path to that file.

If you want to train a model with a VOC dataset, you could run:
- **python train_voc.py --year year**: For example, python train_voc.py --year 2012

If you want to train a model with a COCO dataset, you could run:
- **python train_coco.py --year year**: For example, python train_coco.py --year 2014

If you want to train a model with both COCO datasets (training set = train2014 + val2014 + train2017, val set = val2017), you could run:
- **python train_coco_all.py**

## Test

For each type of dataset (VOC or COCO), I provide 3 different of test scripts:

If you want to test a trained model with a standard VOC dataset, you could run:
- **python test_xxx_dataset.py --year year**: For example, python test_coco_dataset.py --year 2014

If you want to test a model with some images, you could put them into the same folder, whose path is **path/to/input/folder**, then run:
- **python test_xxx_images.py --input path/to/input/folder --output path/to/output/folder**: For example, python train_voc_images.py --input test_images --output test_images

If you want to test a model with a video, you could run :
- **python test_xxx_video.py --input path/to/input/file --output path/to/output/file**: For example, python test_coco_video --input test_videos/input.mp4 --output test_videos/output.mp4

You could find all trained models I have trained in [link](https://drive.google.com/open?id=1gx1qvgu8rZRtEgkCMA9KqJZtFwjr8fc-)

## Experiments:

I run experiments in 2 machines, one with NVIDIA TITAN X 12gb GPU and the other with NVIDIA quadro 6000 24gb GPU.

The training/test loss curves for each experiment are shown below:

-**VOC2007**
-**VOC2012**
-**COCO2014**
-**COCO2014+2017**
You could find detail log of each experiment containing loss, accuracy and confusion matrix at the end of each epoch in **output/datasetname_depth_number/logs.txt**, for example output/ag_news_depth_29/logs.txt
