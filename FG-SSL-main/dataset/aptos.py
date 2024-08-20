import sys
import os

# 将项目的根目录添加到 Python 的路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.setting import jigsaw_generator
from augmentation import *
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List

class barlowDataset(Dataset):
    def __init__(self, args, images, labels, transforms_1=None, transforms_2=None, table_1=None, table_2=None, table_3=None):
        self.args = args

        self.images = images.values
        self.labels = labels.values
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2

        # 在 __init__ 中初始化表格
        self.table_1 = [torch.tensor(args.table[0])] if args.table and len(args.table) > 0 else [torch.tensor(1)]
        self.table_2 = [torch.tensor(args.table[1])] if args.table and len(args.table) > 1 else [torch.tensor(1)]
        self.table_3 = [torch.tensor(args.table[2])] if args.table and len(args.table) > 2 else [torch.tensor(1)]


    def __getitem__(self, idx):
        image_name = self.images[idx]
        img = Image.open(image_name)
        labels = self.labels[idx]
        table_label_0 = random.randrange(0, len(self.table_1))
        table_label_1 = random.randrange(0, len(self.table_2))
        table_label_2 = random.randrange(0, len(self.table_3))

        # 在 __getitem__ 中使用
        table_1 = self.table_1[table_label_0] if len(self.table_1) > 0 else torch.tensor(1)
        table_2 = self.table_2[table_label_1] if len(self.table_2) > 0 else torch.tensor(1)
        table_3 = self.table_3[table_label_2] if len(self.table_3) > 0 else torch.tensor(1)

        if self.transforms_1 is not None and self.transforms_2 is not None:
            img_1 = self.transforms_1(img)   # origin
            img_2 = self.transforms_2(img)   # transform

            # 添加检查确保 n > 0
            if table_1.item() > 0:
                table_1_img = jigsaw_generator(img_1, table_1.type(torch.int))
            else:
                raise ValueError("Table value for n is zero or negative, which is not allowed.")

            if table_2.item() > 0:
                table_2_img = jigsaw_generator(img_1, table_2.type(torch.int))
            else:
                raise ValueError("Table value for n is zero or negative, which is not allowed.")

            if table_3.item() > 0:
                table_3_img = jigsaw_generator(img_1, table_3.type(torch.int))
            else:
                raise ValueError("Table value for n is zero or negative, which is not allowed.")

            return img_1, table_1_img, table_2_img, table_3_img, img_2
        
        else:
            img = self.transforms_1(img)
            return img, labels

    def __len__(self):
        return len(self.images)

    
def aptos_moco_loader(args):
    # 从指定的CSV文件中加载数据
    data_df = pd.read_csv(r"D:\FG-SSL-main\dataset\data\aptos2019-blindness-detection\train.csv")
    images = data_df["id_code"].apply(lambda x: os.path.join(r"D:\FG-SSL-main\dataset\data\aptos2019-blindness-detection\train_images", f"{x}.png"))  # 假设图像文件是 PNG 格式
    labels = data_df["diagnosis"]

    # 定义图像预处理和增强
    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.CenterCrop(args.centercrop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集实例
    dataset = barlowDataset(args, images, labels, transforms_1=transform, transforms_2=transform)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=8)
    
    # 返回三个相同的数据加载器实例，可以根据实际需求调整
    return dataloader, dataloader, dataloader