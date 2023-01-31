import glob
import os
from config import *
import cv2
import numpy as np
import torch
from model import create_model

os.makedirs('..\\test_result\\', exist_ok=True)
# model call
model = create_model(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load('..\\outputs\\model100.pth', map_location=DEVICE))
model.eval()

test_images = glob.glob(os.path.join(VALID_DIR, '*.jpg'))
print(f'Test instances : {len(test_images)}')

detection_threshold = 0.85

for i in range(len(test_images)):
    image_name = test_images[i].split('\\')[-1].split('.')[0]

    # image read
    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)

    image /= 255.0

    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float_)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image)

    # load all detection to cpu for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # print(outputs)
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # item으로 빼도 되지만 numpy로 tensor을 벗길 수 있다
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()

        pred_class = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, pred_class[j], (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('prediction', orig_image)
        cv2.waitKey(0)

        cv2.imwrite(f'..\\test_result\\{image_name}.png', orig_image)
    print(f'Image {i + 1} done ...!')

