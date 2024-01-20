import torch
import torch.nn as nn
# from torchsummary import summary


# 搭建 Incption 模块
class Inception(nn.Module):
    def __init__(self, in_channels, branch_1_out_channels: int, branch_2_out_channels: tuple,
                 branch_3_out_channels: tuple, branch_4_out_channels: int):
        """
        @param in_channels: 输入通道数
        @param branch_1_out_channels: int 第一条线路输出通道数
        @param branch_2_out_channels: tuple 第二条线路输出通道数
        @param branch_3_out_channels: tuple 第三条线路输出通道数
        @param branch_4_out_channels: int 第四条线路输出通道数
        """
        super(Inception, self).__init__()

        # 选择设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # 第一条线路: 1x1 卷积
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_out_channels, kernel_size=1),
            nn.ReLU()
        )

        # 第二条线路: 1x1 卷积 + 3x3 卷积
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_out_channels[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=branch_2_out_channels[0], out_channels=branch_2_out_channels[1], kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        # 第三条线路: 1x1 卷积 + 5x5 卷积
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_out_channels[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=branch_3_out_channels[0], out_channels=branch_3_out_channels[1], kernel_size=5,
                      padding=2),
            nn.ReLU()
        )

        # 第四条线路: 3x3 最大池化 + 1x1 卷积
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_4_out_channels, kernel_size=1),
            nn.ReLU()
        )

        # 前向传播

    def forward(self, x):
        # 将数据加载到模型中
        x = x.to(self.device)

        # 分别计算四条路径的输出
        branch_1_output = self.branch_1(x)
        branch_2_output = self.branch_2(x)
        branch_3_output = self.branch_3(x)
        branch_4_output = self.branch_4(x)

        # 将四条路径的输出拼接在一起, dim = 1 表示在通道维度上拼接
        return torch.cat((branch_1_output, branch_2_output, branch_3_output, branch_4_output), dim=1)


# 搭建 GoogLeNet 模型
class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super(GoogLeNet, self).__init__()

        # 选择设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # 第一个模块: 7x7 卷积 + 3x3 最大池化
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第二个模块: 1x1 卷积 + 3x3 卷积 + 3x3 最大池化
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第三个模块: 两个 Inception 模块 + 3x3 最大池化
        self.block_3 = nn.Sequential(
            Inception(in_channels=192, branch_1_out_channels=64, branch_2_out_channels=(96, 128),
                      branch_3_out_channels=(16, 32), branch_4_out_channels=32),
            Inception(in_channels=256, branch_1_out_channels=128, branch_2_out_channels=(128, 192),
                      branch_3_out_channels=(32, 96), branch_4_out_channels=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第四个模块: 五个 Inception 模块 + 3x3 平均池化
        self.block_4 = nn.Sequential(
            Inception(in_channels=480, branch_1_out_channels=192, branch_2_out_channels=(96, 208),
                      branch_3_out_channels=(16, 48), branch_4_out_channels=64),
            Inception(in_channels=512, branch_1_out_channels=160, branch_2_out_channels=(112, 224),
                      branch_3_out_channels=(24, 64), branch_4_out_channels=64),
            Inception(in_channels=512, branch_1_out_channels=128, branch_2_out_channels=(128, 256),
                      branch_3_out_channels=(24, 64), branch_4_out_channels=64),
            Inception(in_channels=512, branch_1_out_channels=112, branch_2_out_channels=(144, 288),
                      branch_3_out_channels=(32, 64), branch_4_out_channels=64),
            Inception(in_channels=528, branch_1_out_channels=256, branch_2_out_channels=(160, 320),
                      branch_3_out_channels=(32, 128), branch_4_out_channels=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第五个模块: 两个 Inception 模块 + 全局平均池化 + 全连接层
        self.block_5 = nn.Sequential(
            Inception(in_channels=832, branch_1_out_channels=256, branch_2_out_channels=(160, 320),
                      branch_3_out_channels=(32, 128), branch_4_out_channels=128),
            Inception(in_channels=832, branch_1_out_channels=384, branch_2_out_channels=(192, 384),
                      branch_3_out_channels=(48, 128), branch_4_out_channels=128),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=10)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    # 前向传播
    def forward(self, x):
        # 将数据加载到模型中
        x = x.to(self.device)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        return x


# 测试模型
if __name__ == '__main__':
    # 选择设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 创建模型
    model = GoogLeNet(Inception).to(device)

    print(device)

    # # 打印模型结构
    # summary(model, input_size=(3, 224, 224), batch_size=16)
