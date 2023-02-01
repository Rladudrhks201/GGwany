import os
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

file_path = glob.glob(os.path.join('A:\\test01', '*', 'images', '*.jpg'))
print(len(file_path))

for path in file_path:
        img = Image.open(path)
        img = img.resize((960, 540))
        img.save(path)