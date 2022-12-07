import cv2
import numpy as np

# 순서
# get eyes cor > calculate degree > make affine matrix > image affine transform

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

face_img = cv2.imread('./face.png')
face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# 얼굴을 사각형으로 표시

faces = face_cascade.detectMultiScale(face_gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    roi_img = face_img[y:y + h, x:x + w].copy()
    roi_gray = face_gray[y:y + h, x:x + w].copy()

# cv2.imshow('face_box', face_img)
# cv2.waitKey(0)

# 눈을 사각형으로 표시

eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
for (x, y, w, h) in eyes:
    cv2.rectangle(roi_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# cv2.imshow('face_box', roi_img)
# cv2.waitKey(0)
# print(eyes)
eye_1 = eyes[0]
eye_2 = eyes[1]  # 눈의 좌표들을 변수에 저장

# 각 눈의 중심점을 계산
eye_1_center = [int(eye_1[0] + (eye_1[2] / 2)), int(eye_1[1] + (eye_1[3] / 2))]
eye_2_center = [int(eye_2[0] + (eye_2[2] / 2)), int(eye_2[1] + (eye_2[3] / 2))]
eye_1_cx = eye_1_center[0]
eye_1_cy = eye_1_center[1]
eye_2_cx = eye_2_center[0]
eye_2_cy = eye_2_center[1]

cv2.circle(roi_img, (eye_1_cx, eye_1_cy), 5, (255, 0, 0), -1)
cv2.circle(roi_img, (eye_2_cx, eye_2_cy), 5, (0, 255, 0), -1)
cv2.line(roi_img, eye_1_center, eye_2_center, (0, 0, 255), 3)
cv2.imshow("face", roi_img)
cv2.waitKey(0)

# 아크 탄젠트 함수로 눈사이 직선과 수평선의 삼각형의 각도를 구함

tri_width = eye_1_cx - eye_2_cx
tri_height = eye_1_cy - eye_2_cy
angle = np.arctan(tri_height / tri_width)
angle = (angle * 180) / np.pi

# 이미지를 각도 만큼 회전
h, w = roi_img.shape[:2]
center = (w // 2, h // 2)
rotmat = cv2.getRotationMatrix2D(center, (angle), 1.0)
rotated = cv2.warpAffine(roi_img, rotmat, (w, h))
cv2.imshow('face', rotated)
cv2.waitKey(0)

# 남자 사진 얼굴 감지 및 회전
face_img2 = cv2.imread('./face01.png')
face_gray2 = cv2.cvtColor(face_img2, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(face_gray2, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(face_img2, (x, y), (x + w, y + h), (255, 0, 0), 3)


eyes = eye_cascade.detectMultiScale(face_gray2, 1.3, 4)
for (x, y, w, h) in eyes:
    cv2.rectangle(face_img2, (x, y), (x + w, y + h), (0, 255, 0), 3)



cv2.imshow('face_box', face_img2)
cv2.waitKey(0)

eye_1 = eyes[0]
eye_2 = eyes[1]  # 눈의 좌표들을 변수에 저장

# 각 눈의 중심점을 계산
eye_1_center = [int(eye_1[0] + (eye_1[2] / 2)), int(eye_1[1] + (eye_1[3] / 2))]
eye_2_center = [int(eye_2[0] + (eye_2[2] / 2)), int(eye_2[1] + (eye_2[3] / 2))]
eye_1_cx = eye_1_center[0]
eye_1_cy = eye_1_center[1]
eye_2_cx = eye_2_center[0]
eye_2_cy = eye_2_center[1]

cv2.circle(face_img2, (eye_1_cx, eye_1_cy), 5, (255, 0, 0), -1)
cv2.circle(face_img2, (eye_2_cx, eye_2_cy), 5, (0, 255, 0), -1)
cv2.line(face_img2, eye_1_center, eye_2_center, (0, 0, 255), 3)
cv2.imshow("face", face_img2)
cv2.waitKey(0)

# 아크 탄젠트 함수로 눈사이 직선과 수평선의 삼각형의 각도를 구함

tri_width = eye_1_cx - eye_2_cx
tri_height = eye_1_cy - eye_2_cy
angle = np.arctan(tri_height / tri_width)
angle = (angle * 180) / np.pi

# 이미지를 각도 만큼 회전
h, w = face_img2.shape[:2]
center = (w // 2, h // 2)
rotmat = cv2.getRotationMatrix2D(center, (angle), 1.0)
rotated = cv2.warpAffine(face_img2, rotmat, (w, h))
cv2.imshow('face', rotated)
cv2.waitKey(0)
