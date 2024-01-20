# 导入包
import torch
from torch import nn
# from torchsummary import summary


# 定义LeNet-5模型
class LeNet_5(nn.Module):
    # 初始化
    def __init__(self, num_classes=10):
        super(LeNet_5, self).__init__()
        # 决定用 GPU 还是 CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # 定义激活函数
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # 按照LeNet-5网络结构顺序定义网络层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear5 = nn.Linear(5 * 5 * 16, out_features=120)
        self.linear6 = nn.Linear(in_features=120, out_features=84)
        self.linear7 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # 将数据加载到 device 上
        x = x.to(self.device)

        # 第一个卷积块
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool2(x)

        # 第二个卷积块
        x = self.conv3(x)
        x = self.sigmoid(x)
        x = self.pool4(x)

        # 平展后第一个全连接
        x = self.flatten(x)
        x = self.linear5(x)
        x = self.sigmoid(x)

        # 第二个全连接
        x = self.linear6(x)
        x = self.sigmoid(x)

        # 第三个全连接到输出层
        x = self.linear7(x)

        return x


# 主函数
if __name__ == '__main__':
    # 决定使用 GPU 还是 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print('Using device:', device)

    # 实例化模型
    model = LeNet_5().to(device)

    # # 打印模型结构
    # summary(model, input_size=(1, 28, 28), batch_size=128)
