import copy

import albumentations as A
from albumentations.pytorch import ToTensorV2
from custom_dataset import custom_dataset
import matplotlib.pyplot as plt


def aug_function(mode_flag):
    if mode_flag == 'train':
        train_transform = A.Compose([
            A.SmallestMaxSize(max_size=400),
            A.Resize(width=256, height=256),
            A.RandomCrop(height=224, width=224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.06, rotate_limit=20, p=0.8),
            A.RGBShift(r_shift_limit=17, g_shift_limit=17, b_shift_limit=17, p=0.7),
            A.RandomBrightnessContrast(p=0.6),
            A.RandomShadow(p=0.6),
            A.RandomFog(),
            A.RandomShadow(p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
        return train_transform
    elif mode_flag == 'val':
        val_transform = A.Compose([
            A.SmallestMaxSize(max_size=400),
            A.Resize(width=256, height=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(),
            ToTensorV2()
        ])
        return val_transform


def visualize_augmentations(dataset, idx=0, cols=5, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(
        t, (A.Normalize, ToTensorV2)
    )])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]  # 그림 번호
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()

    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     train_transform = aug_function(mode_flag='train')
#     train_dataset = custom_dataset(".\\dataset\\train", transform=train_transform)
#     visualize_augmentations(train_dataset, idx=130)
