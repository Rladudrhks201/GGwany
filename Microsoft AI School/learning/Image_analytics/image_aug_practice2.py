import cv2
import random
import albumentations as A

#   json, xml 둘 중 한가지를 이용하여 동일한 효과를 내는 함수를 만들어보기 !
#   xml -> 커스텀 데이터 셋 -> 첫번째 코드 실습 결과 도출

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_box(image, bboxes, category_ids, category_id_to_name, color=BOX_COLOR, thickness=2):
    # visualize a single bounding box on the image
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        print('class name : ', class_name)
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      color=color, thickness=thickness)

        cv2.putText(img, text=class_name, org=(x_min, y_min - 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=thickness)
    cv2.imshow('test', img)
    cv2.waitKey(0)


image = cv2.imread('C:\\Users\\user\\Desktop\\project\\dogcat\\dogcat.png')

bboxes = [[3.96, 183.38, 200.88, 214.03], [468.94, 92.01, 171.06, 248.45]]
category_ids = [1, 2]
category_id_to_name = {1: 'cat', 2: 'dog'}

# transform
transform = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2),
    A.HorizontalFlip(),
    A.RandomRotate90(),
    # A.MultiplicativeNoise(multiplier=0.5),
    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True)
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
# xml 같은 경우에는 pascal 형식
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

visualize_box(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,
              color=BOX_COLOR, thickness=2)

# visualize_box(image, bboxes, category_ids, category_id_to_name,
#               color=BOX_COLOR, thickness=2)
