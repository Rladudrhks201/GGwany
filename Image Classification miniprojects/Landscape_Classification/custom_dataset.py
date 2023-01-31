import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class custom_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.png'))
        self.transform = transform
        # self.img_list = []  # 이미지를 메모리에 올려서 사용
        # for img_path in self.file_path:
        #     self.img_list.append(Image.open(img_path))
        self.label_dict = {'cloudy': 0, 'desert': 1, 'green_area': 2, 'water': 3}

    def __getitem__(self, index):
        img_path = self.file_path[index]
        img = Image.open(img_path)
        label_temp = img_path.split('\\')[-2]
        label = self.label_dict[label_temp]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.file_path)



