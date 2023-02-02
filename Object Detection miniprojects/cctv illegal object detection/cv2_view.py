import cv2
import json
import os
import glob
path1 = 'A:/Train/images'
path2 = 'A:/Train/labels'

len_ = glob.glob(os.path.join(path1, '*.jpg'))
count = 0
for file_name in os.listdir(path1):
# for i in image_path:
    image = os.path.join(path1, file_name)
    image = cv2.imread(image)
    with open(f'A:/Train/labels/{file_name[:-4]}.json', 'r', encoding='utf8')as f:
        data = json.load(f)

    for j in data['annotations']:
        name = j['object_class']
        x1 = j['bbox'][0][0]
        y1 = j['bbox'][0][1]
        x2 = j['bbox'][1][0]
        y2 = j['bbox'][1][1]

        x1 = int(x1)/2
        y1 = int(y1)/2
        x2 = int(x2)/2
        y2 = int(y2)/2
        
        # print(type(x1), y1, x2, y2)

        cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        cv2.putText(image, name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # image = cv2.resize(image, (960,540))
    cv2.namedWindow(f'{file_name}')
    cv2.moveWindow(f'{file_name}', 800, 400)
    cv2.imshow(f'{file_name}', image)
    waitkey = cv2.waitKey()
    if waitkey == ord('s'):
        print(f"'{file_name},'")
    elif waitkey == ord('p'):
        os.remove(os.path.join(path1, file_name))
        os.remove(f"{path2}/{file_name[:-4]}.json")
        print('삭제 : ',path1, file_name)
    elif waitkey == ord('q'):
        break
    cv2.destroyAllWindows()
    count += 1
    print(f"{count}/{len(len_)}", end='\r')

