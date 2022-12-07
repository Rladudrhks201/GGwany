import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import image_show

img_ori = cv2.imread('./car1.png')
image_show(img_ori)

height, width, channel = img_ori.shape
print(height, width, channel)

# 가우시안
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)

# 이진화
img_blur_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,  # odd over 3
    C=9
)

img_thresh = cv2.adaptiveThreshold(
    img_gray,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,  # odd over 3
    C=9  # 공식에서 사용하는 상수의 값, 거의 양수지만 가끔 0이나 음수도 쓴다
)

plt.figure(figsize=(15, 15))
plt.subplot(1, 5, 1)
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
plt.subplot(1, 5, 2)
plt.imshow(img_gray, cmap='gray')
plt.subplot(1, 5, 3)
plt.imshow(img_blurred, cmap='gray')
plt.subplot(1, 5, 4)
plt.imshow(img_blur_thresh, cmap='gray')
plt.subplot(1, 5, 5)
plt.imshow(img_thresh, cmap='gray')
plt.show()

# Contour , 외곽선찾기

contours, _ = cv2.findContours(
    img_blur_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
# cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(0, 0, 255))
# image_show(temp_result)

contours_dict = []

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    contours_dict.append(
        {
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)  # 중심
        }
    )
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
# cv2.imshow('temp_result', temp_result)
# cv2.waitKey(0)

# 컨투어 박스 생성완료


# 일정 조건을 만족하는 컨투어 박스만 보기
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 최소, 최대 기준의 임시값을 설정후 한번 보기

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']

    if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
for d in possible_contours:
    cv2.rectangle(temp_result, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), color=(0, 255, 0), thickness=2)
cv2.imshow('temp result', temp_result)
cv2.waitKey(0)

# 위 사진은 추려낸 contours들이다.
# 번호판 위치에 contours들이 선별된 걸 볼 수 있지만
# 전혀 관련 없는 영역의 contours들도 저장되었다.
# 이제 더 기준을 강화하여 번호판 글자들을 찾아야한다.


## Select Candidates by Arrangement of Contours

# 남은 contours 중에 확실하게 번호판을 찾기 위해 기준을 강화한다.
# 번호판의 특성을 고려했을 때 세울 수 있는 기준은 아래와 같다.
#
# 1. 번호판 Contours의 width와 height의 비율은 모두 동일하거나 비슷하다.
# 2. 번호판 Contours 사이의 간격은 일정하다.
# 3. 최소 3개 이상 Contours가 인접해 있어야한다. (대한민국 기준)


MAX_DIAG_MULTIPLYER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3

cnt_recursive = 0
def find_chars(contour_list):
    global cnt_recursive
    cnt_recursive += 1
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
        matched_contours_idx.append(d1['idx'])

        # 최소 갯수를 만족할 때 까지 반복
        # 만약 끝까지 갔는데도 못찾으면 for문 완료
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

cv2.imshow("countours box", temp_result)
cv2.waitKey(0)


### Rotate plate image


PLATE_WIDTH_PADDING = 1.3  # 1.3
PLATE_HEIGHT_PADDING = 1.5  # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    print(sorted_chars[0]['cy'])
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )

    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
        0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue

    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    x = int(plate_cx - plate_width / 2)
    y = int(plate_cy - plate_height / 2)
    w = int(plate_width)
    h = int(plate_height)
    print(x, y, w, h)
    num_idx = 1
    for sorted_char in sorted_chars:
        number_crop = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(sorted_char['w']), int(sorted_char['h'])),
            center=(int(sorted_char['cx']), int(sorted_char['cy']))
        )
        ret, number_crop = cv2.threshold(number_crop,127, 255,cv2.THRESH_BINARY_INV)
        # print(number_crop.flatten())
        print(len(np.where(number_crop.flatten() == 255)[0])/len(number_crop.flatten()))
        # print(len(np.where(number_crop.flatten()-127>0)[0]/len(number_crop.flatten())))
        plt.subplot(len(sorted_chars), 2, num_idx)
        num_idx += 1
        plt.imshow(number_crop, 'gray')
        plt.subplot(len(sorted_chars), 2, num_idx)
        plt.hist(number_crop)
        num_idx += 1
    plt.show()
    img_out = img_ori.copy()
    cv2.rectangle(img_out, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
    cv2.imshow("test", img_cropped)
    cv2.imshow("orig", img_out)
    cv2.waitKey(0)