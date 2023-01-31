import torch
from torch.utils.data import Dataset
import os
import glob
import cv2

'''
-Data
    -Product Image
        -Training
            - 각종 제품의 대분류
                - images
        -Validation
            - 각종 제품의 대분류
                - images
'''

class custom_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = glob.glob(os.path.join(file_path, '*', '*.jpg'))
        self.transform = transform
        list_label = os.listdir(file_path)
        list_label2 = []
        for i in list_label:
            if '_' in i:
                list_label2.append(i.split('_')[0])
            else:
                list_label2.append(i)
        list_label2 = list(set(list_label2))
        list_label2.sort()
        self.label_dict = {}
        for index, category in enumerate(list_label2):
            self.label_dict[category] = int(index)
        # print(self.label_dict)

    def __getitem__(self, index):
        img_path = self.file_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_label = img_path.split('\\')[-2]
        if '_' in temp_label:
            temp_label = temp_label.split('_')[0]
        label = self.label_dict[temp_label]
        label = torch.tensor(label)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.file_path)

# {'beverage': 0, 'cannedfood': 1, 'coffeetea': 2, 'dairy': 3, 'dessert': 4, 'drinks': 5, 'hmr': 6, 'noodles': 7, 'quasidrugs': 8, 'sauce': 9, 'snack': 10}
# test = custom_dataset('.\\dataset\\test')
# for i in test:
#     pass