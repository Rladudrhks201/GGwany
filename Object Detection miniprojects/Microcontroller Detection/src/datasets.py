import torch
import cv2
import numpy as np
import os
import glob

import xml.etree.ElementTree as ET
from config import *
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform


# the dataset class
class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transform=None):
        self.dir_path = dir_path
        self.width = width
        self.height = height
        self.classes = classes
        self.transform = transform

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f'{self.dir_path}\\*.jpg')
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, index):
        # capture the image name and the full image path
        image_name = self.all_images[index]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding xml file for getting the annotations
        ann_filename = image_name[:-4] + '.xml'
        ann_filepath = os.path.join(self.dir_path, ann_filename)

        boxes = []
        labels = []
        tree = ET.parse(ann_filepath)
        root = tree.getroot()

        # get the height, width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # xmin
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the adjusted image size
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # bbox to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        image_id = torch.Tensor([index])
        target['image_id'] = image_id

        # apply the image transforms
        if self.transform:
            sample = self.transform(image=image_resized, bboxes=target['boxes'],
                                    labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


# prepare the final datasets and data loaders
train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

# card dataset
card_train_dataset = MicrocontrollerDataset(CARD_TRAIN_DIR, RESIZE_TO, RESIZE_TO, CARD_CLASSES, get_train_transform())
card_valid_dataset = MicrocontrollerDataset(CARD_VALID_DIR, RESIZE_TO, RESIZE_TO, CARD_CLASSES, get_valid_transform())
card_test_dataset = MicrocontrollerDataset(CARD_TEST_DIR, RESIZE_TO, RESIZE_TO, CARD_CLASSES, get_valid_transform())

# dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER,
                          collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER,
                          collate_fn=collate_fn)

# card data loader
card_train_loader = DataLoader(card_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER,
                          collate_fn=collate_fn)
card_valid_loader = DataLoader(card_valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER,
                          collate_fn=collate_fn)
# if __name__ == '__main__':
#     dataset = MicrocontrollerDataset(CARD_TRAIN_DIR, RESIZE_TO, RESIZE_TO, CARD_CLASSES)
#     print(f'Number of Train Images >> {len(card_train_dataset)}')
#
#
#     # function to visualize a single sample
#     def visualize_sample(image, target):
#         box = target['boxes'][0]
#         label = CLASSES[target['labels']]
#
#         cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#
#         cv2.putText(
#             image, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
#         )
#
#         cv2.imshow('show', image)
#         cv2.waitKey(0)
#
#     for i in range(NUM_SAMPLES_TO_VISUALIZE):
#         image, target = dataset[i]
#         visualize_sample(image, target)