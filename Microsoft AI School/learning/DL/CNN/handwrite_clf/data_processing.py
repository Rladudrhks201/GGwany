from utils import image_file, expand2square
import os
from PIL import Image

train_image_path = os.path.join('C:\\Users\\user\\Desktop\\data\\dataset\\', 'train_image\\')
test_image_path = os.path.join('C:\\Users\\user\\Desktop\\data\\dataset\\', 'test_image\\')
train_data = image_file(train_image_path)
test_data = image_file(test_image_path)

# resize
train_image_resize = False
if train_image_resize == True:
    for i in train_data:
        f_name = i.split('\\')[-2]  # 숫자 폴더의 이름
        if f_name == '0':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '1':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '2':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '3':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '4':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '5':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '6':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '7':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '8':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')
        elif f_name == '9':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\train\\{f_name}\\{file_name}.png')

test_image_resize = False
if test_image_resize == True:
    for i in test_data:
        f_name = i.split('\\')[-2]  # 숫자 폴더의 이름
        if f_name == '0':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '1':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '2':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '3':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '4':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '5':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '6':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '7':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '8':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')
        elif f_name == '9':
            img = Image.open(i)
            img_new = expand2square(img, (0, 0, 0)).resize((400, 400))
            os.makedirs(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}', exist_ok=True)
            file_name = os.path.basename(i)
            file_name = file_name.split('.')[0]
            img_new.save(f'C:\\Users\\user\\Desktop\\data\\fnimg\\test\\{f_name}\\{file_name}.png')