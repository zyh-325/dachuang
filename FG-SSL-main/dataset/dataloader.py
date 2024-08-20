from typing import List
from torch.utils.data import DataLoader

from .isiceight import *    # 导入 ISIC 2018 数据集相关的加载器
from .isicseven import *    # 导入 ISIC 2017 数据集相关的加载器
from .aptos import aptos_moco_loader     # 导入 Aptos 数据集相关的加载器


def get_loader(args):
    if args.dataset in ["isic2017", "ISIC-2017", "ISIC2017"]:
        return isicseven_loader(args)      # 加载 ISIC 2017 数据集    
    elif args.dataset in ["isic2018", "ISIC-2018", "ISIC2018"]:
        return isiceight_loader(args)       # 加载 ISIC 2018 数据集
    elif args.dataset in ["aptos", "APTOS", "eyes"]:
        return aptos_moco_loader(args)      # 加载 Aptos 数据集
    
