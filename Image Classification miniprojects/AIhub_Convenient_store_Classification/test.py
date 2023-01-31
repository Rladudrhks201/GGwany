import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
from custom_dataset import custom_dataset
from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
import pandas as pd
import os
import glob
import warnings

net = models.resnet18(pretrained=True)
net.fc = nn.Linear(in_features=512, out_features=11)
print(net)