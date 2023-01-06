import matplotlib.pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split
import cv2


def data_size_show(data_dir='.\\Data'):
    x_plt = []
    y_plt = []
    for directory in os.listdir(data_dir):
        x_plt.append(directory)
        y_plt.append(len(os.listdir(os.path.join(data_dir, directory))))

    # Bar plot
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.barh(x_plt, y_plt, color='maroon')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')  # remove x, y ticks
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    ax.invert_yaxis()  # show top values
    plt.ylabel('RSP Type')
    plt.xlabel('Num of Images')
    plt.title('Rock Scissors Paper')
    plt.show()



os.makedirs('.\\dataset\\train', exist_ok=True)
os.makedirs('.\\dataset\\test', exist_ok=True)
os.makedirs('.\\dataset\\val', exist_ok=True)


def split_data(species, file_path='.\\Data'):
    y_temp = []
    data_path = glob.glob(os.path.join(file_path, f'{species}', '*.png'))
    for path in data_path:
        y_data = path.split('\\')[-2]
        y_temp.append(y_data)
    x_tr_temp, x_tp_temp, y_tr_temp, y_tp_temp = train_test_split(data_path, y_temp, train_size=0.8,
                                                                  random_state=2324)
    x_val_temp, x_te_temp, y_val_temp, y_te_temp = train_test_split(x_tp_temp, y_tp_temp, train_size=0.5,
                                                                    random_state=2324)
    for path, label in zip(x_tr_temp, y_tr_temp):
        os.makedirs(f'.\\dataset\\train\\{label}', exist_ok=True)
        img = cv2.imread(path)
        cv2.imwrite(f'.\\dataset\\train\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

    for path, label in zip(x_val_temp, y_val_temp):
        os.makedirs(f'.\\dataset\\val\\{label}', exist_ok=True)
        img = cv2.imread(path)
        cv2.imwrite(f'.\\dataset\\val\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

    for path, label in zip(x_te_temp, y_te_temp):
        os.makedirs(f'.\\dataset\\test\\{label}', exist_ok=True)
        img = cv2.imread(path)
        cv2.imwrite(f'.\\dataset\\test\\{label}\\{os.path.basename(path).split(".")[0]}.png', img)

RSP_list = os.listdir('.\\Data')

# train, val, test data 생성
# for i in RSP_list:
#     split_data(i)

# data_size_show()
print('전체 종의 개수 >>', len(RSP_list))
num_tr = len(glob.glob(os.path.join('.\\dataset\\train', '*', '*.png')))
num_val = len(glob.glob(os.path.join('.\\dataset\\val', '*', '*.png')))
num_te = len(glob.glob(os.path.join('.\\dataset\\test', '*', '*.png')))
print('train len >> ', num_tr)
print('val len >> ', num_val)
print('test len >> ', num_te)