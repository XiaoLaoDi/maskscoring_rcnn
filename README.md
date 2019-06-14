Mask Scoring R-CNN (MS R-CNN)
-----------------
Forking from https://github.com/zjhuang22/maskscoring_rcnn

Introduction
-----------------
[Mask Scoring R-CNN](https://arxiv.org/pdf/1903.00241.pdf) contains a network block to learn the quality of the predicted instance masks. The proposed network block takes the instance feature and the corresponding predicted mask together to regress the mask IoU. The mask scoring strategy calibrates the misalignment between mask quality and mask score, and improves instance segmentation performance by prioritizing more accurate mask predictions during COCO AP evaluation. By extensive evaluations on the COCO dataset, Mask Scoring R-CNN brings consistent and noticeable gain with different models and different frameworks. The network of MS R-CNN is as follows:

![alt text](demo/network.png)


Install
-----------------
  Check [INSTALL.md](INSTALL.md) for installation instructions.


Prepare Data
----------------
```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
  ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
  ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
 ```
coco dataset can be download from here [link](http://cocodataset.org/#download) or (thanks to) [link](https://blog.csdn.net/u014734886/article/details/78830713)

coco dataset description can be found in COCO official website [link](http://cocodataset.org/#home) or in [link](https://zhuanlan.zhihu.com/p/29393415)


Pretrained instance segmentation Models
---------------
```
  STEP ONE: mkdir models
  STEP TWO: download my coco2017 pretrained instance segmentation models
  STEP THREE: put it in directory ---> models
```
My coco2017 training log and pre-trained models can be found here [link](链接:https://pan.baidu.com/s/14UqK2I1yZN1ykoxocJX0gw)(pw:53hd).


Testing Pretrained Models
---------------
```
  STEP ONE: download pretrained instance segmentation Models
  STEP TWO: python demo/demo.py
```
![alt text](demo/result.png)


