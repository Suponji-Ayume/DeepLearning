import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from tqdm import tqdm
# 导入模型
from model import LeNet5


# 处理数据集，划分为训练集和验证集
def test_data_process(dataset, resize: tuple, batch_size=1, shuffle=True, num_workers=4):
    # 下载数据集
    test_data = dataset(root='../datasets',
                        train=False,
                        transform=transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]),
                        download=True)

    # 将训练集和验证集转换为可迭代的 DataLoader 对象
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers)
    return test_dataloader


# 测试模型
def test_model(model, test_dataloader, show_detail=False):
    # 决定使用 GPU 还是 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # 将模型加载到 device 当中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_sample_num = 0.0
    test_acc = 0.0

    # 只进行前向传播，不计算梯度，从而节省内存，加快计算速度
    with torch.no_grad():
        with tqdm(total=len(test_dataloader), colour="green", ncols=135,
                  unit=' batch') as pbar:
            # 设置进度条的前缀
            pbar.set_description('Testing: ')
            pbar.set_postfix_str(
                "Test Accuracy: {:.2f}%".format(0))

            # 遍历测试集
            for batch_images, batch_labels in test_dataloader:
                # 将数据加载到 device 上
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                # 将模型设置为评估模式
                model.eval()

                # 前向传播
                output = model(batch_images)
                # 计算预测值
                predict_label = torch.argmax(output, dim=1)
                # 更新预测正确的样本数
                test_corrects += torch.sum(torch.eq(predict_label, batch_labels)).item()
                # 更新预测的样本数
                test_sample_num += batch_labels.size(0)

                # 显示预测结果
                if (show_detail == True):
                    predict_result = predict_label.item()
                    true_result = batch_labels.item()
                    print('Predict Label: {} | True Label: {}'.format(predict_result, true_result))

                # 更新进度条的检测量
                pbar.set_postfix_str(
                    "Test Accuracy: {:.2f}%".format(round(test_corrects / test_sample_num, 4) * 100))
                # 更新进度条
                pbar.update(1)
    # 计算准确率
    test_acc = test_corrects / test_sample_num
    test_acc = round(test_acc, 4)

    # 打印准确率，格式化为百分比形式
    print('Test Accuracy: {:.2%}'.format(test_acc))


if __name__ == '__main__':
    # 实例化模型
    model = LeNet5()
    # 单机多卡验证
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # 加载模型参数
    model.load_state_dict(torch.load('../output/FashionMNIST/best_model.pth'))
    # 处理数据集
    test_dataloader = test_data_process(FashionMNIST, resize=(28, 28), batch_size=1, shuffle=True, num_workers=4)
    # 测试模型
    test_model(model, test_dataloader, show_detail=False)
