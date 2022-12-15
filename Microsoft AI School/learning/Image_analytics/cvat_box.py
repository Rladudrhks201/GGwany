import os
import glob
import cv2
from xml.etree.ElementTree import parse


# xml 파일 찾을 수 있는 함수 제작
def find_xml_file(xml_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(xml_folder_path):
        for filename in files:
            # image.xml -> .xml
            ext = os.path.splitext(filename)[-1]
            if ext == '.xml':
                root = os.path.join(path, filename)
                # ./cvat_annotations/annotations.xml
                all_root.append(root)
            else:
                break
    return all_root


xml_dirs = find_xml_file('/learning/image01/cvat_annotations/')
# print(xml_dirs)

for xml_dir in xml_dirs:
    tree = parse(xml_dir)
    root = tree.getroot()
    img_metas = root.findall('image')
    for img_meta in img_metas:
        # xml에 기록된 이미지 이름, 사이즈 정보
        image_name = img_meta.attrib['name']
        image_height = int(img_meta.attrib['height'])
        image_width = int(img_meta.attrib['width'])
        image_path = os.path.join('/learning/image01/images',
                                  image_name)
        image = cv2.imread(image_path)

        # box meta info
        box_metas = img_meta.findall('box')
        for box_meta in box_metas:
            box_label = box_meta.attrib['label']
            box = [
                int(float(box_meta.attrib['xtl'])),
                int(float(box_meta.attrib['ytl'])),
                int(float(box_meta.attrib['xbr'])),
                int(float(box_meta.attrib['ybr']))
            ]
            # print(box[0:4])

            rect_img = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 225, 255), 2)
        cv2.imshow('test', rect_img)
        cv2.waitKey(0)
