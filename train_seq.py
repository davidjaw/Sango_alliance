import cv2
import numpy as np
from typing import List, Tuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models, datasets
from torchvision.transforms import Compose, ToTensor, RandomAffine, Resize, Normalize
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import itertools
import matplotlib.pyplot as plt
import random
import yaml
import torchvision
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from data_loader import CustomOCRDataset, prepare_data_transform


def main():
    train_transform = prepare_data_transform()
    valid_transform = prepare_data_transform(mode='test')
    post_transform = lambda x: F.interpolate(x.unsqueeze(0), size=(16*3, 130*3), mode='nearest').squeeze(0)
    train_dataset = CustomOCRDataset(1e3, transform=train_transform, post_transform=post_transform)
    valid_dataset = CustomOCRDataset(1e3, transform=valid_transform, post_transform=post_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=30, shuffle=True, num_workers=4)


if __name__ == '__main__':
    main()
