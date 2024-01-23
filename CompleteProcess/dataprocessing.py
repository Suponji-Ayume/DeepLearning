import os
import shutil

import numpy as np
from PIL import Image


# 将测试集中的图片按照类别分类，并分别保存到不同的文件夹中
def classify_train_data(dataset):
    # 输入文件夹和输出文件夹的路径
    input_folder = "../Datasets/{}/train".format(dataset)
    output_folder = "../Datasets/{}/train".format(dataset)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):  # 确保文件是图片文件
            # 获取图片名称和前缀
            image_name = os.path.splitext(filename)[0]
            prefix = image_name.split(".")[0]

            # 创建目标子文件夹（如果不存在）
            target_folder = os.path.join(output_folder, prefix)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # 构建新的文件路径和文件名
            new_filename = os.path.join(target_folder, image_name.split('.')[1] + ".jpg")

            if not os.path.exists(new_filename):
                # 移动文件到目标子文件夹并重命名
                shutil.move(os.path.join(input_folder, filename), new_filename)

    print("Classify train data successfully!")


# 计算所有三通道图片的每一个通道的均值和方差
def normalize(dataset):
    """
    @param dataset: 要计算训练集图片均值和方差的数据集
    @return: 均值和方差
    """
    # 全部训练集的路径
    data_path = "../Datasets/{}/train".format(dataset)

    # 累积变量
    total_pixels = 0
    sum_normalized_pixels_values = np.zeros(3)


    # 遍历文件夹中的图片文件
    for path, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith((".jpg", ".png", ".bmp", ".jpeg")):
                # 获取图片路径
                image_path = os.path.join(path, filename)
                # 打开图片
                image = Image.open(image_path)
                # 将图片转换为数组
                image_array = np.array(image)

                # 将像素值归一化到 0-1 之间
                normalized_pixels_values = image_array / 255.0
                # 累积归一化的像素值和像素数量
                sum_normalized_pixels_values += normalized_pixels_values.sum(axis=(0, 1))
                total_pixels += normalized_pixels_values.size

    # 计算均值
    mean = sum_normalized_pixels_values / total_pixels

    # 计算标准差
    sum_diff_square = np.zeros(3)
    # 遍历文件夹中的图片文件
    for path, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith((".jpg", ".png", ".bmp", ".jpeg")):
                # 获取图片路径
                image_path = os.path.join(path, filename)
                # 打开图片
                image = Image.open(image_path)
                # 将图片转换为数组
                image_array = np.array(image)

                # 将像素值归一化到 0-1 之间
                normalized_pixels_values = image_array / 255.0
                # 计算方差
                sum_diff_square += ((normalized_pixels_values - mean) ** 2).sum(axis=(0, 1))
    std = np.sqrt(sum_diff_square / total_pixels)

    return mean, std

if __name__ == '__main__':
    # 计算训练集图片的均值和方差
    mean, std = normalize("Dogs_VS_Cats")
    print("Mean: ", mean)
    print("Std: ", std)