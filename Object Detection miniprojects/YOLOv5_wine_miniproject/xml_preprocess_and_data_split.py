import glob
import os
import xml.etree.ElementTree as ET
import cv2
import shutil
from sklearn.model_selection import train_test_split

# label dict
classes = ['AlcoholPercentage', 'Appellation AOC DOC AVARegion', 'Appellation QualityLevel',
           'CountryCountry', 'Distinct Logo', 'Established YearYear', 'Maker-Name', 'Organic',
           'Sustainable', 'Sweetness-Brut-SecSweetness-Brut-Sec', 'TypeWine Type', 'VintageYear']
label_dict = {k: v for v, k in enumerate(classes)}

# folder create
os.makedirs('.\\dataset\\train', exist_ok=True)
os.makedirs('.\\dataset\\train\\images', exist_ok=True)
os.makedirs('.\\dataset\\train\\labels', exist_ok=True)
os.makedirs('.\\dataset\\test', exist_ok=True)
os.makedirs('.\\dataset\\test\\images', exist_ok=True)
os.makedirs('.\\dataset\\test\\labels', exist_ok=True)
os.makedirs('.\\dataset\\valid', exist_ok=True)
os.makedirs('.\\dataset\\valid\\images', exist_ok=True)
os.makedirs('.\\dataset\\valid\\labels', exist_ok=True)
os.makedirs('.\\dataset\\temp', exist_ok=True)

# xml, jpg path
xml_path = glob.glob(os.path.join('.\\dataset\\wine labels_voc_dataset', '*.xml'))


# VOC -> YOLO format
def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]



for path in xml_path:
    name_temp = path.split('\\')[-1].split('.xml')[0]
    image_temp = name_temp + '.jpg'
    image_path = os.path.join('.\\dataset\\wine labels_voc_dataset', image_temp)
    image = cv2.imread(image_path)
    w = image.shape[1]
    h = image.shape[0]
    tree = ET.parse(path)
    root = tree.getroot()
    boxes = []
    labels = []


    for member in root.findall('object'):
        try:
            label = member.find('name').text
            label = label_dict[label]
        except:
            print(member.find('name').text)
            exit()
        xmin = int(member.find('bndbox').find('xmin').text)
        xmax = int(member.find('bndbox').find('xmax').text)
        ymin = int(member.find('bndbox').find('ymin').text)
        ymax = int(member.find('bndbox').find('ymax').text)
        bbox = [xmin, ymin, xmax, ymax]
        bbox = xml_to_yolo_bbox(bbox, w, h)
        yolo_x = str(round(bbox[0], 3))
        yolo_y = str(round(bbox[1], 3))
        yolo_w = str(round(bbox[2], 3))
        yolo_h = str(round(bbox[3], 3))


        with open(f'.\\dataset\\temp\\{name_temp + ".txt"}', 'a') as f:
            f.write(f'{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n')
    if os.path.exists(f'.\\dataset\\temp\\{name_temp + ".txt"}') == False:
        with open(f'.\\dataset\\temp\\{name_temp + ".txt"}', 'w') as f:
            f.write('')

    seen_count += 1



# data split

y_temp = []
jpg_path = []
for path in xml_path:
    y_data = path.split('\\')[-1].split('.xml')[0] + '.txt'
    y_temp.append(os.path.join('.\\dataset\\temp\\', y_data))
    jpg = path.split('\\')[-1].replace('.xml', '.jpg')
    jpg_path.append(os.path.join('.\\dataset\\wine labels_voc_dataset\\', jpg))

x_tr_temp, x_tp_temp, y_tr_temp, y_tp_temp = train_test_split(jpg_path, y_temp, train_size=0.8,
                                                              random_state=2324)
x_val_temp, x_te_temp, y_val_temp, y_te_temp = train_test_split(x_tp_temp, y_tp_temp, train_size=0.5,
                                                                random_state=2324)
for path, label in zip(x_tr_temp, y_tr_temp):
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\train\\images\\{os.path.basename(path)}', img)
    shutil.move(label, f'.\\dataset\\train\\labels\\{os.path.basename(label)}')

for path, label in zip(x_val_temp, y_val_temp):
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\valid\\images\\{os.path.basename(path)}', img)
    shutil.move(label, f'.\\dataset\\valid\\labels\\{os.path.basename(label)}')

for path, label in zip(x_te_temp, y_te_temp):
    img = cv2.imread(path)
    cv2.imwrite(f'.\\dataset\\test\\images\\{os.path.basename(path)}', img)
    shutil.move(label, f'.\\dataset\\test\\labels\\{os.path.basename(label)}')
