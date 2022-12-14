"""
dataset
    - train
        - cat
        - dog
    - val
        - cat
        - dog
    - test
        - cat
        - dog
"""
from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image
import cv2

label_dict = {'cat': 0, 'dog': 1}


class cat_dog_mycustomdataset(Dataset):
    def __init__(self, data_path):
        # csv folder 읽기, 변환 할당, 데이터 필터링 등과 같은 초기 논리 발생
        # data_path <- 'C:\\Users\\user\\Desktop\\project\\dataset\\train'
        self.all_data_path = glob.glob(os.path.join(data_path, '*', '*.jpg'))
        # print(self.all_data_path)
        # 하위 폴더의 모든 폴더내의 jpg 파일을 가져옴

    def __getitem__(self, index):
        image_path = self.all_data_path[index]
        # print(image_path)
        img = Image.open(image_path).convert('RGB')
        label_temp = image_path.split('\\')
        # print(label_temp[-1][:3])
        label = label_dict[label_temp[-1][:3]]
        print(img, label)
        return img, label
    def __len__(self):
        # 전체 데이터 길이 반환
        return len(self.all_data_path)


test = cat_dog_mycustomdataset('C:\\Users\\user\\Desktop\\project\\dataset\\train\\')
for i in test:
    pass
