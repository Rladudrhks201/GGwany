import os
import glob
from torch.utils.data import Dataset
import cv2


class Custom_Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.png'))
        self.labels_list = os.listdir(file_path)
        self.labels_list.sort()
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.file_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_label = img_path.split('\\')[-2]
        label = self.labels_list.index(temp_label)
        label = int(label)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

    def __len__(self):
        return len(self.file_path)
