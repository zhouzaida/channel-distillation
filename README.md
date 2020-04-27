# Channel Distillation

## Novelty

1. distill channel features
2. distillation loss rate decay
3. improved knowledge distillation loss

## Installion

## Training

Running the following command and experiment will be launched.

```bash
python3 ./train.py
```

If you want to run other experiments, you just need modify following losses in config.py

+ s_resnet18.t_resnet34.cd.ce
```python
loss_list = [
    {"loss_name": "CELoss", "loss_rate": 1, "factor": 1, "loss_type": "ce_family", "loss_rate_decay": "lrdv1"},
    {"loss_name": "CDLoss", "loss_rate": 6, "factor": 1, "loss_type": "fd_family", "loss_rate_decay": "lrdv1"},
]
```

+ s_resnet18.t_resnet34.cd.ce.lrdv2
```python
loss_list = [
    {"loss_name": "CELoss", "loss_rate": 1, "factor": 1, "loss_type": "ce_family", "loss_rate_decay": "lrdv2"},
    {"loss_name": "CDLoss", "loss_rate": 6, "factor": 0.9, "loss_type": "fd_family", "loss_rate_decay": "lrdv2"},
]
```

+ s_resnet18.t_resnet34.cd.kdv2.lrdv2
```python
loss_list = [
    {"loss_name": "CELoss", "loss_rate": 1, "factor": 1, "loss_type": "ce_family", "loss_rate_decay": "lrdv2"},
    {"loss_name": "KDLossv2", "T": 1, "loss_rate": 1, "factor": 1, "loss_type": "kdv2_family", "loss_rate_decay": "lrdv2"},
    {"loss_name": "CDLoss", "loss_rate": 6, "factor": 0.9, "loss_type": "fd_family", "loss_rate_decay": "lrdv2"},
]
```

## Experiments

+ dataset
  
  imagenet

+ model
  
  + student: resnet18
  + teacher: resnet34

+ loss detail

+ result

| Model structure | Testing on validation| Top-1 error | Top-5 error | Reference |
| -- | -- | -- | -- | -- |
| ResNet18-B | 1-crop | 30.43 | 10.76 | [ResNet-Github](https://github.com/facebook/fb.resnet.torch) |
| ResNet18.at | 1-crop | 29.30 | 10.00 | [attention-transfer](https://github.com/szagoruyko/attention-transfer) |
| s_resnet18.t_resnet34.kd.ce | 1-crop | 29.50 | 9.52 | KD |
| s_resnet18.t_resnet34.cd.ce | 1-crop | 28.38 | 9.48 | Ours |
| s_resnet18.t_resnet34.cd.ce.lrdv1 | 1-crop | 28.36(28.34) | 9.39(9.41) | ZZD(ZGCR) |
| s_resnet18.t_resnet34.cd.ce.lrdv2 | 1-crop | 28.09(28.00) | 9.34(9.27) | ZZD(ZGCR) |
| s_resnet18.t_resnet34.cd.kd.lrdv2 | 1-crop | 27.99 | 9.31 | Ours |
| s_resnet18.t_resnet34.cd.kdv2.lrdv2 | 1-crop | 27.73(27.68) | 9.22(9.39) |ZZD(ZGCR) |

## Reference

+ [SCA-CNN: Spatial and Channel-Wise Attention in Convolutional Networks for Image Captioning](https://ieeexplore.ieee.org/document/8100150)
+ [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)