import random
import torch
import os
import glob
from torch.utils.data import Dataset
import cv2
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


class custom_dataset(Dataset):
    def __init__(self, file_path):
        self.filepath = glob.glob(os.path.join(file_path, '*', '*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image_path = self.filepath[index]
        img = Image.open(image_path).convert('RGB')
        label = os.path.dirname(image_path).split('\\')[-1]
        label = int(label)
        # print(image_path, label)
        mo = os.path.dirname(image_path).split('\\')[-2]
        if mo == 'train':
            if random.uniform(0, 1) < 0.2 or img.getbands()[0] == 'L':  # L 모드 : 255 gray scale
                # random gray scale from 20%
                img = img.convert('L').convert('RGB')

            if random.uniform(0, 1) < 0.2:
                # random gaussian blur from 20%
                gaussianBlur = ImageFilter.GaussianBlur(random.uniform(0.5, 1.2))
                img = img.filter(gaussianBlur)

            else:
                if img.getbands()[0] == 'L':
                    img = img.convert('L').convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filepath)

# 시각화 테스트 !
# train_dataset = custom_dataset('C:\\Users\\labadmin\\Desktop\\data\\fnimg\\train')
#
# _, ax = plt.subplots(2, 4, figsize=(16, 10))
# for i in range(8):
#     data = train_dataset.__getitem__(np.random.choice(range(train_dataset.__len__())))
#
#     image = data[0].cpu().detach().numpy().transpose(1, 2, 0) * 255
#     imag = image.astype(np.uint32)
#
#     label = data[1]
#
#     ax[i // 4][i - (i // 4) * 4].imshow(image.astype("uint8"))
#     ax[i // 4][i - (i // 4) * 4].set_title(label)
#
# plt.show()
