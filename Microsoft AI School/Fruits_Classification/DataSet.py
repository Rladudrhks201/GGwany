from torch.utils.data import Dataset
import cv2
import os
import glob
from PIL import Image

label_dict = {"dekopon": 0,  "orange": 1,  "kanpei": 2, "kiwi": 3}

class custom_dataset(Dataset):
    def __init__(self, image_file_path, transform=None, transform2=None):
        self.image_file_path = glob.glob(os.path.join(image_file_path, '*', '*.png'))
        self.transform = transform
        self.transform2 = transform2

    def __getitem__(self, index):
        image_path = self.image_file_path[index]    # index 값은 순서대로가 아니라 랜덤값이다 (1번씩은 꼭 나오게 나옴)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_temp = os.path.basename(image_path).split(' (')[0]
        label = label_dict[label_temp]
        if self.transform:
            img = self.transform(image=image)['image']
        if self.transform2:
            img2 = self.transform2(image=image)['image']
        img = img.float()
        # return img, label   # 학습용
        return img, label, image_path   #테스트용
    def __len__(self):
        return len(self.image_file_path)


if __name__ == '__main__':
    test = custom_dataset('C:\\Users\\user\\Desktop\\Search\\Data\\train', transform=None)
    for i in test:
        pass
