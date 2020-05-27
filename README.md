# Channel Distillation
PyTorch implement of "Channel Distillation: Channel-Wise Attention for Knowledge Distillation"

## Novelty

1. Channel Distillation
2. Guided Knowledge Distillation
3. Early Decay Teacher

<center>
<img src="./assets/arch.png" width="60%" height="60%" />

Distillation Network Architecture
</center>

## Requirements

> Python >= 3.7  
>PyTorch >= 1.2.0

## Training

### Prepare ImageNet Dataset

### Running Experiments

Running the following command and experiment will be launched.

```bash
python3 ./train.py
```

If you want to run other experiments, you just need modify following losses in `config.py`

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

+ Dataset
  
  ImageNet

+ Model
  
  + Student: ResNet18
  + Teacher: ResNet34

+ Result

| Model structure | Testing on validation| Top-1 error | Top-5 error | Reference |
| -- | -- | -- | -- | -- |
| ResNet18-B | 1-crop | 30.43 | 10.76 | [ResNet-Github](https://github.com/facebook/fb.resnet.torch) |
| ResNet18.at | 1-crop | 29.30 | 10.00 | [attention-transfer](https://github.com/szagoruyko/attention-transfer) |
| s_resnet18.t_resnet34.kd.ce | 1-crop | 29.50 | 9.52 | KD |
| s_resnet18.t_resnet34.cd.ce | 1-crop | 28.38 | 9.48 | Ours |
| s_resnet18.t_resnet34.cd.ce.lrdv1 | 1-crop | 28.34 | 9.41 | Ours |
| s_resnet18.t_resnet34.cd.ce.lrdv2 | 1-crop | 28.00 | 9.27 | Ours |
| s_resnet18.t_resnet34.cd.kd.lrdv2 | 1-crop | 27.99 | 9.31 | Ours |
| s_resnet18.t_resnet34.cd.kdv2.lrdv2 | 1-crop | 27.68 | 9.39 | Ours |

+ Comparion result with other methods

| Method | Model | Top-1 acc(%) | Top-5 acc(%) | Reference |
| -- | -- | -- | -- | -- |
| Teacher | ResNet34 | 73.30 | 91.42 | [PyTorch](https://pytorch.org/hub/pytorch_vision_resnet/) |
| Student | ResNet18 | 69.76 | 89.08 | [PyTorch](https://pytorch.org/hub/pytorch_vision_resnet/) |
| KD | ResNet18 | 70.50 | 90.48 | Ours |
| FitNets | ResNet18 | 70.66 | 89.23 | [Residual Knowledge Distillation](https://arxiv.org/abs/2002.09168) 
| AT | ResNet18 | 70.70 | 90.00 |[attention-transfer](https://github.com/szagoruyko/attention-transfer) |
| RKD + AT | ResNet18 | 71.54 | 90.26 | [Residual Knowledge Distillation](https://arxiv.org/abs/2002.09168) | 
| **CD(Ours)** | ResNet18 | 72.32 | 91.61 | Ours |