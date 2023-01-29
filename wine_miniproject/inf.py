import torch
import cv2
import xml.etree.ElementTree as ET
import glob
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./runs/train/exp0023/weights/best.pt')
# print(model)

# Inference Settings
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IOU threshold

# device settings
# model.cuda()    # model.cpu()는 cpu
model.to(device)

# label dict
classes = ['AlcoholPercentage', 'Appellation AOC DOC AVARegion', 'Appellation QualityLevel',
           'CountryCountry', 'Distinct Logo', 'Established YearYear', 'Maker-Name', 'Organic',
           'Sustainable', 'Sweetness-Brut-SecSweetness-Brut-Sec', 'TypeWine Type', 'VintageYear']
label_dict = {k: v for k, v in enumerate(classes)}

# image
image_path = glob.glob(os.path.join('.\\dataset\\test\\images', '*.jpg'))
# xml write
tree = ET.ElementTree()
root = ET.Element('annotations')

seen_count = 0
for path in image_path:
    name_temp = path.split('\\')[-1].split('.jpg')[0]
    image_temp = os.path.basename(path)
    # image read
    img = cv2.imread(path)
    # image size h w c
    h, w, c = img.shape

    xml_frame = ET.SubElement(root, "image", id=str(seen_count),
                              width=str(w), height=str(h), name=str(image_temp))

    # Inference
    result = model(img, size=640)
    # print(result.print())
    # print(result.xyxy)  # bbox


    bbox = result.xyxy[0]
    for bbox_info in bbox:
        # print(bbox_info)
        x1 = str(round(bbox_info[0].item(),6))
        y1 = str(round(bbox_info[1].item(),6))
        x2 = str(round(bbox_info[2].item(),6))
        y2 = str(round(bbox_info[3].item(),6))
        sc = round(bbox_info[4].item(),6)
        label_number = bbox_info[5].item()
        label = label_dict[int(label_number)]
        # print(x1, y1, x2, y2, sc, label_number)
    
        
        if sc >= model.conf:
            ET.SubElement(xml_frame, 'box', z_order='0', occluded='0', source='manual',
                        xtl=x1, ytl=y1, xbr=x2, ybr=y2, label=label)
    
        # xyxy to yolo center_x, center_y, w, h
        # center_x = round(((x1 + x2) / 2) / w, 6)
        # center_y = round(((y1 + y2) / 2) / h, 6)
        # yolo_w = round((x2 - x1) / w, 6)
        # yolo_h = round((y2 - y1) / h, 6)
        # print(center_x, center_y, yolo_w, yolo_h)
    
        # yolo center_X, center_y, w, h save
        # with open(f'./adit_mp4-85_jpg.rf.349f560cdca44171ecd778c8a889c5c4.txt' , 'a') as f:
        #     f.write(f'{int(label_number)} {center_x} {center_y} {yolo_w} {yolo_h}\n')
    
    
        # img1 = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        # ret = cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)



    # cv2.imshow('test', ret)
    # cv2.waitKey(0)


    seen_count += 1

os.makedirs('.\\runs\\train\\exp0023\\VOCXML', exist_ok=True)
tree._setroot(root)
tree.write('.\\runs\\train\\exp0023\\VOCXML\\yolov5m_ann.xml', encoding='utf-8')