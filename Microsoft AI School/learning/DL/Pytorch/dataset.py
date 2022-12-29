import glob
import os.path
import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms


label_dic = {
    'cat': 0, 'dog': 1
}


class MyCustomDataset(Dataset):
    def __init__(self, path):
        # Data path
        all_data_path = glob.glob(os.path.join(..))
        self.transforms = transforms
        pass

    def __getitem__(self, index):
        image_path = all_data_path[index]
        label_temp = image_path.split('\\')
        label_temp = label_temp[2]
        label_temp = label_temp.replace('.jpg', '')
        label = label_dic[label_temp]
        image = cv2.imread(image_path)


        # image augmentation
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label
        pass

    def __len__(self):
        pass
