import cv2

def imshow(image):
    cv2.imshow('show', image)
    cv2.waitKey(0)

image = cv2.imread('./images (2).jpeg')

# 이미지 블러
# blur() : 각 픽셀에 커널 개수의 역수를 곱하여 모두 더함
image_blurry = cv2.blur(image, (5,5)) # 5x5 커널 평균 값으로 이미지를 흐리게함, 7 이상은 식별이 힘들수도 있음
imshow(image_blurry)