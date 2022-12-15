import numpy as np
import random
import cv2
import albumentations
from albumentations.pytorch import ToTensorV2

keypoints = [

    (100, 100, 50, np.pi / 4.0),

    (720, 410, 50, np.pi / 4.0),

    (1100, 400, 50, np.pi / 4.0),

    (1700, 30, 50, np.pi / 4.0),

    (300, 650, 50, np.pi / 4.0),

    (1570, 590, 50, np.pi / 4.0),

    (560, 800, 50, np.pi / 4.0),

    (1300, 750, 50, np.pi / 4.0),

    (900, 1000, 50, np.pi / 4.0),

    (910, 780, 50, np.pi / 4.0),

    (670, 670, 50, np.pi / 4.0),

    (830, 670, 50, np.pi / 4.0),

    (1000, 670, 50, np.pi / 4.0),

    (1150, 670, 50, np.pi / 4.0),

    (820, 900, 50, np.pi / 4.0),

    (1000, 900, 50, np.pi / 4.0),

]

Keypoint_COLOR = (0, 255, 0)  # GREEN


def vis_keypoints(image, keypoints, color=Keypoint_COLOR, diameter=15):
    image = image.copy()
    for (x, y, s, a) in keypoints:
        print(x, y, s, a)
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)

        x0 = int(x) + s * np.cos(a)
        y0 = int(y) - s * np.sin(a)
        cv2.arrowedLine(image, (int(x), int(y)), (int(x0), int(y0)), color, 2)

    cv2.imshow('test', image)
    cv2.waitKey(0)


image = cv2.imread('C:\\Users\\user\\Desktop\\project\\catcat\\fox.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = albumentations.Compose([
    albumentations.RandomCrop(224, 224),
    albumentations.ShiftScaleRotate(p=1),
    # ToTensorV2()
], keypoint_params=albumentations.KeypointParams(format='xysa', angle_in_degrees=False))
# 사이즈가 변경되어도 점 위치가 안바뀜
transformed = transform(image=image, keypoints=keypoints)

# vis_keypoints(image, keypoints)
vis_keypoints(transformed['image'], transformed['keypoints'])
