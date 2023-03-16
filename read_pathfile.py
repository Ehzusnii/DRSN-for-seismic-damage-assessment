#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import pickle

def read_path_label(root: str):
    """
    对图片数据集进行分割
    :param root: 数据集所在的路径(不同类型图片所在文件夹路径)
    :param val_rate: 验证集在数据集中所占的比例
    :return: 训练图片路径，训练图片标签，验证集图片路径，验证集图片标签
    """
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root)
                    if os.path.isdir(os.path.join(root, cla))]  # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(root+'/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []      # 存储训练集的所有图片路径
    train_images_label = []     # 存储训练集图片对应索引信息

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG",".npz"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))

    return train_images_path, train_images_label

