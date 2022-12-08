import cv2

large_img = cv2.imread('./tennis.png')
watermark = cv2.imread('./goat.png')
small_img = cv2.resize(watermark, (300, 300))

x_offset = 400
y_offset = 170
rows, columns, channels = small_img.shape
roi = large_img[y_offset:470, x_offset:700]

# logo image 빨간색 부분을 제외한 모든 것을 필터링 하도록 -> 회색 이미지로 변경
small_img_gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(small_img_gray, 120, 255, cv2.THRESH_BINARY)

bg = cv2.bitwise_or(roi, roi, mask=mask)
# cv2.imshow('test', bg)
# cv2.waitKey(0)

# 마스크를 다시 빨간색 워트마크로 전환시키려면 마스크 반전이 필요함
mask_inv = cv2.bitwise_not(mask)
fg = cv2.bitwise_and(small_img, small_img, mask=mask_inv)
# cv2.imshow('test', fg)
# cv2.waitKey(0)

final_roi = cv2.add(bg, fg)
# cv2.imshow('final', final_roi)
# cv2.waitKey(0)

large_img[y_offset:470, x_offset:700] = final_roi
cv2.imshow('show', large_img)
cv2.waitKey(0)