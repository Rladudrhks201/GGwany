import cv2
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision import transforms


class Alb_aug_dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image

    def __len__(self):
        return len(self.file_paths)


alb_transform = A.Compose([
    A.Resize(256, 256),
    A.Cutout(num_holes=20, max_h_size=15, max_w_size=15,p=1),
    ToTensorV2()
])
file_paths = ['C:\\Users\\user\\Desktop\\project\\fruits1\\fruit.png']
alb_dataset = Alb_aug_dataset(file_paths, transform=alb_transform)

for i in range(1):
    sample = alb_dataset[0]

plt.figure(figsize=(10, 10))
plt.imshow(transforms.ToPILImage()(sample))
plt.show()
