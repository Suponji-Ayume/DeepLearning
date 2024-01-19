from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# 下载数据集
train_dataset = MNIST(root='./data', train=True,
                      transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]), download=True)
# 加载数据集
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)

# 读取一个batch的数据
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
class_labels = train_dataset.classes
print('Class labels:', class_labels)

# 可视化一个batch的数据
images, labels = next(iter(train_loader))
images = images.numpy()
labels = labels.numpy()
fig = plt.figure(figsize=(25, 25))
for idx in np.arange(64):
    ax = fig.add_subplot(8, 64 // 8, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(class_labels[labels[idx]])
plt.show()
