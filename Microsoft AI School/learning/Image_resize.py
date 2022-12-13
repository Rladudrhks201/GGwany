import os
from PIL import Image


def expand2sqare(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        # image add (추가 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


img = Image.open('C:\\Users\\user\\Desktop\\project\\image\\flower.png')
# print(img)
# 결과 > <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=174x290 at 0x1572C970C70>
img_new = expand2sqare(img, (0, 0, 0)).resize((224, 224))
img_new.save('C:\\Users\\user\\Desktop\\project\\image\\flower_new.png')

