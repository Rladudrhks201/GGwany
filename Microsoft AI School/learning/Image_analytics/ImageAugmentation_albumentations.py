import time

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import cv2
import albumentations
from albumentations.pytorch import ToTensorV2


# albumentations Data Pipeline

class alb_CatDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = glob.glob(os.path.join(file_paths, '*'))
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        # Read an image with cv2
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # transform time check
        start_time = time.time()
        if self.transform is not None:
            image = self.transform(image=image)['image']
        end_time = (time.time() - start_time)

        return image, end_time

    def __len__(self):
        return len(self.file_paths)


# transforms
albumentation_transform = albumentations.Compose([
    # albumentations.resize(256, 256),
    # albumentations.RandomCrop(224, 224),
    albumentations.VerticalFlip(),
    albumentations.HorizontalFlip(),
    ToTensorV2()
])

albumentation_transform_oneof = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.RandomCrop(224, 224),
    albumentations.OneOf([
        albumentations.HorizontalFlip(p=1),
        albumentations.RandomRotate90(p=1),
        albumentations.VerticalFlip(p=1)
    ], p=1),
    albumentations.OneOf([
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)
    ], p=1),
    ToTensorV2()
])  # Oneof 메서드는 리스트 안의 한가지 변형만 선택하는 메서드

cat_dataset = alb_CatDataset(file_paths='C:\\Users\\user\\Desktop\\project\\catcat\\',
                             transform=albumentation_transform)
cat_dataset2 = alb_CatDataset(file_paths='C:\\Users\\user\\Desktop\\project\\catcat\\',
                             transform=albumentation_transform_oneof)


total_time = 0
for i in range(100):
    image, end_time = cat_dataset2[0]
    total_time += end_time

print("albumentation time/image >> ", total_time * 10)

plt.Figure(figsize=(10, 10))
plt.imshow(transforms.ToPILImage()(image).convert('RGB'))
plt.show()

# 시간차이가 엄청나게 발생, torchvision의 최적화문제
