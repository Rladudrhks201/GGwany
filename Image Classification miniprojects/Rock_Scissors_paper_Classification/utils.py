import sys
from timm.loss import LabelSmoothingCrossEntropy
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from Custom_dataset import custom_dataset, game_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import os
import cv2
import random


def expand2sqare(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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
    # net = models.mobilenet_v3_small(pretrained=True)
    # net.classifier[3] = nn.Linear(in_features=1024, out_features=3, bias=True)
    # net.to(device)

    # resnet18
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(in_features=512, out_features=3)
    net.to(device)

    model_path = '.\\models\\best.pt'
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    correct = 0
    total = 0
    criterion = LabelSmoothingCrossEntropy()
    test_loss = 0
    test_steps = len(test_loader)
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout, colour='blue')
        for data in test_bar:
            image, label = data
            images, labels = image.to(device), label.to(device)
            output = net(images)
            test_loss += criterion(output, labels).item()
            _, argmax = torch.max(output, 1)
            total += images.size(0)
            correct += (labels == argmax).sum().item()

        acc = correct / total * 100
        loss = test_loss / test_steps
        print(f'Test Loss >> {loss}', f'Test acc >> {acc}%')


def game(file_path='.\\images'):  # 저장된 이미지를 통해 가위바위보 시뮬레이션
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(),
        ToTensorV2()
    ])
    test_dataset = game_dataset(file_path, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mobilenet_v3_small
    net = models.mobilenet_v3_small(pretrained=True)
    net.classifier[3] = nn.Linear(in_features=1024, out_features=3, bias=True)
    net.to(device)

    model_path = '.\\models\\best.pt'
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    with torch.no_grad():
        total_win = 0
        total_loss = 0
        total_draw = 0
        for i, (image) in enumerate(test_loader):
            images = image.to(device)
            output = net(images)
            argmax = torch.argmax(output, dim=1)
            user = int(argmax)
            if user == 0:
                if random.choice([0, 1, 2]) == 0:
                    print("둘 다 보자기이므로 무승부")
                    total_draw += 1
                elif random.choice([0, 1, 2]) == 1:
                    print("나는 보자기를 내고, 컴퓨터는 주먹이므로 승리 !")
                    total_win += 1
                else:
                    print("나는 보자기를 내고 컴퓨터는 가위이므로 패배..")
                    total_loss += 1
            if user == 1:
                if random.choice([0, 1, 2]) == 2:
                    print("나는 주먹을 내고 컴퓨터는 가위를 냈으므로 승리 !")
                    total_win += 1
                elif random.choice([0, 1, 2]) == 0:
                    print("나는 주먹을 내고 컴퓨터는 보자기를 냈으므로 패배..")
                    total_loss += 1
                else:
                    print("둘 다 주먹이므로 무승부")
                    total_draw += 1
            if user == 2:
                if random.choice([0, 1, 2]) == 0:
                    print("나는 가위를 내고 컴퓨터는 보자기를 냈으므로 승리 !")
                    total_win += 1
                elif random.choice([0, 1, 2]) == 1:
                    print("나는 가위를 내고 컴퓨터는 주먹을 냈으므로 패배..")
                    total_loss += 1
                else:
                    print("둘 다 가위이므로 무승부")
                    total_draw += 1
    print('총 게임 수는 >> ', len(test_dataset))
    print('총 승리 수는 >> ', total_win)
    print('총 무승부 수는 >> ', total_draw)
    print('총 패배 수는 >> ', total_loss)


def open_webcam():  # 웹캠을 통해 이미지를 저장
    os.makedirs("./images", exist_ok=True)
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    sample_num = 0
    captured_num = 0

    # loop through frames
    while webcam.isOpened():

        # read frame from webcam
        status, frame = webcam.read()
        sample_num = sample_num + 1

        if not status:
            break

        # display output

        cv2.imshow("captured frames", frame)

        if sample_num == 4:
            captured_num = captured_num + 1

            cv2.imwrite('./images/img' + str(captured_num) + '.jpg', frame)
            sample_num = 0

        print(sample_num)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources
    webcam.release()
    cv2.destroyAllWindows()
