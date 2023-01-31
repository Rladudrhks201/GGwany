import os
import glob
import shutil

# test_path = glob.glob(os.path.join('.\\dataset\\UTKFace', '*.jpg'))
# print(len(test_path))
# os.makedirs('.\\testdata', exist_ok=True)

# for path in test_path:
#     file_name = os.path.basename(path)
#     age = int(file_name.split('_')[0])
#     gender = file_name.split('_')[1]
#     race = file_name.split('_')[2]
#     if (age >=20) and (race == '2'):
#         shutil.move(path, os.path.join('.\\testdata', f'{file_name}'))

test_path = glob.glob(os.path.join('.\\testdata', '*.jpg'))
print(len(test_path))

for path in test_path:
    file_name = os.path.basename(path)
    age = int(file_name.split('_')[0])
    gender = file_name.split('_')[1]
    ages = (age // 10) * 10
    if (age >=60):
        ages = 60
    shutil.move(path, os.path.join('.\\testdata', f'{file_name}'))