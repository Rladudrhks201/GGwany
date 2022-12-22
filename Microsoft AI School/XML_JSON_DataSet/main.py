import cv2
import numpy as np
from torch.utils.data import Dataset
from xml.etree.ElementTree import parse
import json


class customDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        if data_path.split('.')[-1] == 'xml':
            self.dataType = 'xml'
            tree = parse(self.data_path)
            root = tree.getroot()
            self.img_metas = root.findall("image")
            self.dataSize = len(self.img_metas)
        else:
            self.dataType = 'json'
            with open(self.data_path, "r") as f:
                info = json.load(f)
            assert len(info) > 0, "파일 읽기 실패"
            self.categories = dict()
            for category in info['categories']:
                self.categories[category["id"]] = category["name"]
            self.image_infos = info['annotations']

    def __getitem__(self, index):
        if self.dataType == 'xml':
            img_meta = self.img_metas[index]
            image_name = img_meta.attrib['name']
            # box meta info
            box_metas = img_meta.findall("box")
            labels = []
            bboxes = []
            for box_meta in box_metas:
                labels.append(box_meta.attrib['label'])
                bboxes.append([
                    int(float(box_meta.attrib['xtl'])),
                    int(float(box_meta.attrib['ytl'])),
                    int(float(box_meta.attrib['xbr'])),
                    int(float(box_meta.attrib['ybr']))
                ])
            return image_name, labels, bboxes
        else:
            image_info = self.image_infos[index]
            return self.categories[image_info['category_id']], image_info['bbox']

    def __len__(self):
        return self.dataSize


def cvTest(image, dataset, target_size=None):

    img = image.copy()
    y_ = image.shape[0]
    x_ = image.shape[1]
    if target_size:
        x_scale = target_size / x_
        y_scale = target_size / y_
        # print("x_scle >> ", x_scale, "y_scle >> ", y_scale)

        img = cv2.resize(image, (target_size, target_size))
    if dataset.dataType == 'xml':
        data = dataset[0]
        labels = data[1]
        bboxes = data[2]
        for i, box in enumerate(bboxes):
            label = labels[i]
            (x_min, y_min, x_max, y_max) = box
            # xywh to xyxy
            x1, x2, y1, y2 = int(x_min), int(x_max), int(y_min), int(y_max)
            if target_size:
                x1 = int(np.round(x1 * x_scale))
                y1 = int(np.round(y1 * y_scale))
                x2 = int(np.round(x2 * x_scale))
                y2 = int(np.round(y2 * y_scale))
            cv2.putText(img, text=label, org=(x1, y1 - 15),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0),
                        thickness=1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    else:
        for data in dataset:
            label, (x_min, y_min, w, h) = data
            # xywh to xyxy
            x1, x2, y1, y2 = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
            if target_size:
                x1 = int(np.round(x1 * x_scale))
                y1 = int(np.round(y1 * y_scale))
                x2 = int(np.round(x2 * x_scale))
                y2 = int(np.round(y2 * y_scale))

            cv2.putText(img, text=label, org=(x1, y1 - 15),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0),
                        thickness=1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imshow(dataset.dataType + str(target_size), img)
    cv2.imwrite(f'./{dataset.dataType}_{str(target_size)}.png', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    xmlDataset = customDataset('C:\\Users\\user\\Desktop\\project\\xml_folder\\annotations.xml')
    jsonDataset = customDataset('C:\\Users\\user\\Desktop\\project\\xml_folder\\instances_default.json')

    image = cv2.imread('C:\\Users\\user\\Desktop\\project\\xml_folder\\01.png')

    cvTest(image, xmlDataset)
    cvTest(image, jsonDataset)
    cvTest(image, xmlDataset, 400)
    cvTest(image, jsonDataset, 400)
