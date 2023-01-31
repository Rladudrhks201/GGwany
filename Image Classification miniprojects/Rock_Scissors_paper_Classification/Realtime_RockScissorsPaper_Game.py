import numpy as np
import torch
import torch.nn
import torchvision
from torch.autograd import Variable
import PIL
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
import torch.nn as nn
from Custom_dataset import game_dataset
from utils import *

# This is the Label
Labels = {'paper': 0, 'rock': 1, 'scissors': 2}

# Let's preprocess the inputted frame

data_transforms = A.Compose([
    A.SmallestMaxSize(max_size=160),
    A.Resize(width=224, height=224),
    A.Normalize(),
    ToTensorV2()
])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")  ##Assigning the Device which will do the calculation

# model = models.mobilenet_v3_small(pretrained=True)
# model.classifier[3] = nn.Linear(in_features=1024, out_features=3, bias=True)
# model.to(device)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=3)
model.to(device)

model_path = '.\\models\\best.pt'
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()  # set the device to eval() mode for testing


# Set the Webcam
def Webcam_720p():
    cap.set(3, 1280)
    cap.set(4, 720)


def RSP(prediction):
    argmax = torch.argmax(prediction, dim=1)
    user = int(argmax)
    if user == 0:
        if random.choice([0, 1, 2]) == 0:
            print("둘 다 보자기이므로 무승부")
        elif random.choice([0, 1, 2]) == 1:
            print("나는 보자기를 내고, 컴퓨터는 주먹이므로 승리 !")

        else:
            print("나는 보자기를 내고 컴퓨터는 가위이므로 패배..")

    if user == 1:
        if random.choice([0, 1, 2]) == 2:
            print("나는 주먹을 내고 컴퓨터는 가위를 냈으므로 승리 !")

        elif random.choice([0, 1, 2]) == 0:
            print("나는 주먹을 내고 컴퓨터는 보자기를 냈으므로 패배..")

        else:
            print("둘 다 주먹이므로 무승부")

    if user == 2:
        if random.choice([0, 1, 2]) == 0:
            print("나는 가위를 내고 컴퓨터는 보자기를 냈으므로 승리 !")

        elif random.choice([0, 1, 2]) == 1:
            print("나는 가위를 내고 컴퓨터는 주먹을 냈으므로 패배..")

        else:
            print("둘 다 가위이므로 무승부")


def preprocess(image):
    # Webcam frames are numpy array format

    image = data_transforms(image=image)['image']
    image = image.float()

    # image = image.cuda()
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only

    return image  # dimension out of our 3-D vector Tensor


# Let's start the real-time classification process!

cap = cv2.VideoCapture(0)  # Set the webcam
Webcam_720p()

fps = 0
sequence = 0

while True:
    ret, frame = cap.read()  # Capture each frame

    if fps == 4:
        # image = frame[100:450, 150:570]
        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = preprocess(image)
        # print(image_data)
        prediction = model(image_data)
        RSP(prediction)
        fps = 0

    fps += 1
    cv2.imshow("ASL SIGN DETECTER", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("ASL SIGN DETECTER")
