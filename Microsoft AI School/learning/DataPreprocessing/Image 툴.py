from PIL import Image
import cv2

# opencv -> BGR
# Image -> RGB

img = cv2.imread('./images.jpeg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb_img)
pil_img.show()