import torch
from torch.utils.data import Dataset
import os
import glob
import cv2



class custom_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.jpg'))
        self.transform = transform
        list_label = os.listdir(file_path)
        list_label.sort()
        self.label_dict = {}
        for index, category in enumerate(list_label):
            self.label_dict[category] = int(index)
        # print(self.label_dict)

    def __getitem__(self, index):
        img_path = self.file_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_label = img_path.split('\\')[-2]
        label = self.label_dict[temp_label]
        label = torch.tensor(label)

        if self.transform:
            img = self.transform(image=img)['image']
        # print(img, label)
        return img, label

    def __len__(self):
        return len(self.file_path)

# label_dict >>
# {'20_0': 0, '20_1': 1, '30_0': 2, '30_1': 3, '40_0': 4, '40_1': 5, '50_0': 6, '50_1': 7, '60_0': 8, '60_1': 9}

test = custom_dataset('.\\dataset\\dataset\\val')
for i in test:
    pass

