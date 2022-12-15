from PIL import Image
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import os


# 기존 torchvision Data pipeline
# 1. dataset class -> image loader -> transform
class CatDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = glob.glob(os.path.join(file_paths, '*'))
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        # Read an image with PIL
        image = Image.open(file_path)

        # transform time check
        start_time = time.time()
        if self.transform:
            image = self.transform(image)
        end_time = (time.time() - start_time)

        return image, end_time

    def __len__(self):
        return len(self.file_paths)


#### data augmentation transforms
# train, val 따로 생성하여 사용, test 값은 val의 transform을 활용
torchvision_transform = transforms.Compose([
    # transforms.Pad(padding=10),  # 패딩
    # transforms.Resize((224, 224)),  # 리사이즈
    # transforms.CenterCrop(size=30),  # 가운데 위치를 자름
    # transforms.Grayscale(),  # 흑백
    # transforms.ColorJitter(brightness=0.2, contrast=0.3),  # 이미지 밝기, 대조 변경
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
    # transforms.RandomPerspective(distortion_scale=0.7, p=1.0),  # 이미지를 다른 각도에서 보는 것처럼 수정
    # transforms.RandomRotation(degrees=(0, 100)),    # 이미지를 회전시킴
    # transforms.RandomAffine(
    #     degrees=(30, 60), translate=(0.1, 0.3), scale=(0.5, 0.7)
    # ),  # 랜덤한 어파인변환시킴, translate는 가로, 세로, scale은 크기 범위
    # transforms.ElasticTransform(alpha=220.0),   # 쭈글쭈글 이상하게 변함
    transforms.RandomHorizontalFlip(),  # 좌우
    transforms.RandomVerticalFlip(),    # 상하
    # transforms.AutoAugment(),   # 랜덤 변환
    transforms.ToTensor()
])
# 여러 변환을 묶어서 함

# 'C:\\Users\\user\\Desktop\\project\\catcat\\cat.png'
cat_dataset = CatDataset(file_paths='C:\\Users\\user\\Desktop\\project\\catcat\\',
                         transform=torchvision_transform)

total_time = 0
for i in range(100):    # 100번동안 변환하는데 걸리는 시간
    image, end_time = cat_dataset[0]
    total_time += end_time

print("torchvision time/image >> ", total_time * 10)

plt.title('image')
plt.imshow(transforms.ToPILImage()(image).convert('RGB'))
plt.show()
