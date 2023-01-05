import copy
import sys

from custom_dataset import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn


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


def test():
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(),
        ToTensorV2()
    ])
    test_dataset = custom_dataset('.\\dataset\\test', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # swin_b
    # net = models.swin_b(weights='IMAGENET1K_V1')
    # net.head = nn.Linear(in_features=1024, out_features=450, bias=True)
    # net.to(device)

    # resnet50
    # net = models.resnet50(pretrained=True)
    # net.fc = nn.Linear(in_features=2048, out_features=450)
    # net.to(device)

    # Mobilenet_v3_small
    net = models.mobilenet_v3_small(pretrained=True)
    net.classifier[3] = nn.Linear(in_features=1024, out_features=450, bias=True)
    net.to(device)


    model_path = '.\\models\\best.pt'
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout, colour='blue')
        for data in test_bar:
            image, label = data
            images, labels = image.to(device), label.to(device)
            output = net(images)
            _, argmax = torch.max(output, 1)
            total += images.size(0)
            correct += (labels == argmax).sum().item()

        acc = correct / total
        print(f'Test acc >> {acc}%')


