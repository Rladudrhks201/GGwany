import cv2
import random
import albumentations as A


def visualize(image):
    cv2.imshow('viaualization', image)
    cv2.waitKey(0)


# Load the image from the disk
image = cv2.imread('C:\\Users\\user\\Desktop\\project\\road\\road.png')

# visualize the image
# visualize(image)

# RandomRain
transform = A.Compose([
    # A.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=5, p=1)
    # A.RandomSnow(brightness_coeff=2.5, snow_point_upper=0.4, snow_point_lower=0.3, p=1)
    # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.3) # 햇빛번짐
    # A.RandomShadow(num_shadows_lower=2, num_shadows_upper=2, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1))
    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.4, alpha_coef=0.08, p=1)
])
transformed = transform(image=image)
visualize(transformed['image'])  # 비오는 상황 시각화
