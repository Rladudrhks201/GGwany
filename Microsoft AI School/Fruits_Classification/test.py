import torch
from torchvision import transforms
from torchvision import models
from DataSet import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def acc_function(correct, total):
    acc = correct / total * 100
    return acc


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target, path) in enumerate(test_loader):
            image_path = path[0]
            # img = cv2.imread(image_path)
            img = data[0]
            img = (img.detach() * 255).byte().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.uint8).copy()



            file_name = os.path.basename(image_path)
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, argmax = torch.max(output, 1)
            total += target.size(0)
            correct += (target == argmax).sum().item()

            argmax_temp = argmax.item()
            cv2.putText(img, str(argmax_temp), (90, 90), cv2.FONT_ITALIC, 1, (0, 0, 0), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.imwrite(
                f'C:\\Users\\user\\Desktop\\1222_img_aug_test\\after_aug\\{file_name}_actual{target[0]}_pred{argmax_temp }.png',
                img)


        acc = acc_function(correct, total)
        print("acc for {} image : {:.2f}%".format(total, acc))


def main():
    test_transform = A.Compose([
        A.Resize(height=224, width=224),
        # A.HorizontalFlip(p=1),
        A.Cutout(num_holes=20, max_h_size=15, max_w_size=15, p=1),
        ToTensorV2()
    ])

    test_transform2 = A.Compose([
        A.Resize(height=224, width=224),
        A.Cutout(num_holes=20, max_h_size=15, max_w_size=15, p=1)
    ])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.__dict__['resnet18'](pretrained=False, num_classes=4)
    net = net.to(device)

    net.load_state_dict(torch.load('./model/30.pt', map_location=device))  # 학습환경에 상관없게 현재 device 환경으로 적용
    test_data = custom_dataset("C:\\Users\\user\\Desktop\\Search\\Data\\val", transform=test_transform,
                               transform2=test_transform2)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    test(net, test_loader, device)


if __name__ == "__main__":
    main()
