import torch.nn as nn

from .resnet import resnet18, resnet34, resnet50, resnet152


def conv1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class ChannelDistillResNet1834(nn.Module):

    def __init__(self, num_classes=1000, dataset_type="imagenet"):
        super().__init__()
        self.student = resnet18(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)
        self.teacher = resnet34(True, num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)

        self.s_t_pair = [(64, 64), (128, 128), (256, 256), (512, 512)]
        self.connector = nn.ModuleList([conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x)
        ts = self.teacher(x)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillResNet50152(nn.Module):
    def __init__(self, num_classes=100, dataset_type="imagenet"):
        super().__init__()
        self.student = resnet50(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)
        self.teacher = resnet152(True, num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)

        self.s_t_pair = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x)
        ts = self.teacher(x)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts
