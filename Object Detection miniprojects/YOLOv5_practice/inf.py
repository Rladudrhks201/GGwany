import torch
import cv2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./runs/train/exp2/weights/best.pt')
# print(model)

# Inference Settings
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IOU threshold

# device settings
# model.cuda()    # model.cpu()는 cpu
model.to(device)

# one image
image_path = './dataset/test/images/adit_mp4-85_jpg.rf.349f560cdca44171ecd778c8a889c5c4.jpg'

# image read
img = cv2.imread(image_path)

# label dict
label_dict = {0: 'big bus', 1: 'big truck', 2: 'bus-l-', 3: 'bus-s-', 4: 'car', 5: 'mid truck', 6: 'small bus',
              7: 'small truck', 8: 'truck-l-', 9: 'truck-m-', 10: 'truck-s-', 11: 'truck-xl-'}

# Inference
result = model(img, size=640)
# print(result.print())
# print(result.xyxy)  # bbox
bbox = result.xyxy[0]
for bbox_info in bbox:
    # print(bbox_info)
    x1 = bbox_info[0].item()
    y1 = bbox_info[1].item()
    x2 = bbox_info[2].item()
    y2 = bbox_info[3].item()
    sc = bbox_info[4].item()
    label_number = bbox_info[5].item()
    label = label_dict[int(label_number)]
    # print(x1, y1, x2, y2, sc, label_number)

    # image size h w c
    h, w, c = img.shape

    # xyxy to yolo center_x, center_y, w, h
    center_x = round(((x1 + x2) / 2) / w, 6)
    center_y = round(((y1 + y2) / 2) / h, 6)
    yolo_w = round((x2 - x1) / w, 6)
    yolo_h = round((y2 - y1) / h, 6)
    # print(center_x, center_y, yolo_w, yolo_h)

    # yolo center_X, center_y, w, h save
    with open(f'./adit_mp4-85_jpg.rf.349f560cdca44171ecd778c8a889c5c4.txt' , 'a') as f:
        f.write(f'{int(label_number)} {center_x} {center_y} {yolo_w} {yolo_h}\n')


#     img1 = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
#     ret = cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#
# cv2.imshow('test', ret)
# cv2.waitKey(0)
