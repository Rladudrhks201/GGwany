import cv2

# 동영상 속성 확인
cap = cv2.VideoCapture('./video01.mp4')
# cap = cv2.VideoCapture(0)  # 웹캠 사용
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_counter = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

print("width: ", width, "height: ", height)
print('fps: ', fps, 'frame_count: ', frame_counter)

"""
width:  1280.0 height:  720.0
fps:  25.0 frame_count:  323.0
"""

# 동영상 파일 읽기
if cap.isOpened():  # 캡쳐 객체 초기화 확인
    while True:
        ret, frame = cap.read()  # ret은 제대로 읽었는지 여부, frame은 프레임
        if ret:
            cv2.imshow('video file show', frame)
            cv2.waitKey(25)  # 25프레임 영상이라
        else:
            break
else:
    print('영상 불러오기 실패!')
