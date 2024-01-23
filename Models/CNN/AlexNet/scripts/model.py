import torch
import torch.nn as nn
import torch.nn.functional as F


# 搭建 AlexNet 网络模型
class AlexNet(nn.Module):
    # 初始化网络结构
    def __init__(self, input_channels = 1, num_classes = 10):
        super(AlexNet, self).__init__()

        # 选择设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # 定义激活函数层
        self.ReLU = nn.ReLU()

        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=11, stride=4, padding=0)
        # 第二层池化层
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 第四层池化层
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # 第五层卷积层
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 第六层卷积层
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 第七层卷积层
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 第八层池化层
        self.pool8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # 平展层
        self.flatten = nn.Flatten()
        # 第九层全连接层
        self.fc9 = nn.Linear(in_features=9216, out_features=4096)
        # 第十层全连接层
        self.fc10 = nn.Linear(in_features=4096, out_features=4096)
        # 第十一层全连接层
        self.fc11 = nn.Linear(in_features=4096, out_features=num_classes)

    # 前向传播
    def forward(self, x: torch.Tensor):
        # 将数据加载到设备上
        x = x.to(self.device)

        x = self.ReLU(self.conv1(x))
        x = self.pool2(x)
        x = self.ReLU(self.conv3(x))
        x = self.pool4(x)
        x = self.ReLU(self.conv5(x))
        x = self.ReLU(self.conv6(x))
        x = self.ReLU(self.conv7(x))
        x = self.pool8(x)

        x = self.flatten(x)  # 平展

        x = self.ReLU(self.fc9(x))
        x = F.dropout(x, p=0.5)
        x = self.ReLU(self.fc10(x))
        x = F.dropout(x, p=0.5)
        x = self.fc11(x)

        return x


if __name__ == '__main__':
    # 选择设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 实例化网络
    model = AlexNet().to(device)
