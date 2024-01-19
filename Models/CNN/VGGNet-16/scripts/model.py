import torch
import torch.nn as nn
# from torchsummary import summary

# 定义 VGGNet-16 模型
class VGGNet_16(nn.Module):
    def __init__(self):
        super(VGGNet_16, self).__init__()

        # 选择设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # 第一个卷积块
        self.conv_block_1 = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第二个卷积层
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二个卷积块
        self.conv_block_2 = nn.Sequential(
            # 第三个卷积层
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第四个卷积层
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三个卷积块
        self.conv_block_3 = nn.Sequential(
            # 第五个卷积层
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第六个卷积层
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第七个卷积层
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第四个卷积块
        self.conv_block_4 = nn.Sequential(
            # 第八个卷积层
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第九个卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第十个卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第五个卷积块
        self.conv_block_5 = nn.Sequential(
            # 第十一个卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第十二个卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 第十三个卷积层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 激活函数
            nn.ReLU(),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接块
        self.fc_block = nn.Sequential(
            # 平展层
            nn.Flatten(),
            # 第一个全连接层
            nn.Linear(512 * 7 * 7, 4096),
            # # 激活函数
            # nn.ReLU(),
            # # Dropout
            # nn.Dropout(p=0.5),
            # 第二个全连接层
            nn.Linear(4096, 4096),
            # # 激活函数
            # nn.ReLU(),
            # # Dropout
            # nn.Dropout(p=0.5),
            # 第三个全连接层
            nn.Linear(4096, 10)
        )

        # 权重初始化
        for m in self.modules():
            # 判断是否为卷积层
            if isinstance(m, nn.Conv2d):
                # 使用kaiming初始化方法
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # 初始化偏置
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 判断是否为全连接层
            elif isinstance(m, nn.Linear):
                # 使用正态分布初始化方法
                nn.init.normal_(m.weight, 0,0.001)
                # 初始化偏置
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # 前向传播
    def forward(self,x):
        # 将数据加载到设备上
        x = x.to(self.device)

        # 第一个卷积块
        x = self.conv_block_1(x)
        # 第二个卷积块
        x = self.conv_block_2(x)
        # 第三个卷积块
        x = self.conv_block_3(x)
        # 第四个卷积块
        x = self.conv_block_4(x)
        # 第五个卷积块
        x = self.conv_block_5(x)
        # 全连接块
        x = self.fc_block(x)

        return x

if __name__ == "__main__":

    # 选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 创建模型并加载到设备上
    model = VGGNet_16().to(device)

    # # 打印模型
    # summary(model, (1, 224, 224),batch_size=1024)
