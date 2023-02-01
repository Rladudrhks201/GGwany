import os
import glob
import shutil

def rename_legal():
    folder_name = {'18.위해방지시설': 'fence', '19.벤치': 'bench', '20.공원화분': 'park_pot', '21.쓰레기통': 'trash_can',
                   '22.휴게공간': 'rest_area', '23.화장실': 'toilet', '24.비석': 'park_headstone',
                   '25.가로등': 'street_lamp', '26.공원안내표지판': 'park_info'}
    legal_paths = [r'A:/test01/Train/images/train_data/legal',
                   r'A:/test01/Train/labels/train_data/legal',
                   r'A:/test01/Valid/images/valid_data/legal',
                   r'A:/test01/Valid/labels/valid_data/legal']

    for legal_path in legal_paths:
        for name in os.listdir(legal_path):
            original_path = os.path.join(legal_path, name)
            try:
                rename_path = os.path.join(legal_path, folder_name[name])
                os.rename(original_path, rename_path)
            except:
                pass

def rename_illegal():
    folder_name = {'9.쓰레기봉투': 'garbage_bag', '10.좌판': 'sit_board', '11.노점상': 'street_vendor',
                   '12.푸드트럭': 'food_truck', '13.현수막': 'banner', '14.텐트': 'tent', '15.연기': 'smoke',
                   '16.불꽃': 'flame', '17.반려동물': 'pet'}
    illegal_paths = [r'A:/test01/Train/images/train_data/illegal',
                   r'A:/test01/Train/labels/train_data/illegal',
                   r'A:/test01/Valid/images/valid_data/illegal',
                   r'A:/test01/Valid/labels/valid_data/illegal']

    for illegal_path in illegal_paths:
        for name in os.listdir(illegal_path):
            original_path = os.path.join(illegal_path, name)
            try:
                rename_path = os.path.join(illegal_path, folder_name[name])
                os.rename(original_path, rename_path)
            except:
                pass

def restruct_folder():
    legal_paths = [r'A:/test01/Train/images/train_data/legal',
                   r'A:/test01/Train/labels/train_data/legal',
                   r'A:/test01/Valid/images/valid_data/legal',
                   r'A:/test01/Valid/labels/valid_data/legal']

    illegal_paths = [r'A:/test01/Train/images/train_data/illegal',
                     r'A:/test01/Train/labels/train_data/illegal',
                     r'A:/test01/Valid/images/valid_data/illegal',
                     r'A:/test01/Valid/labels/valid_data/illegal']

    for legal_path in legal_paths:
        tr_vl = legal_path.split('/')[-4]
        type = legal_path.split('/')[-3]
        files_list = glob.glob(os.path.join(legal_path, '*', '*'))
        for file in files_list:
            fname = os.path.basename(file)
            shutil.move(file, f'A:\\test01\\{tr_vl}\\{type}\\{fname}')
    for illegal_path in illegal_paths:
        tr_vl = illegal_path.split('/')[-4]
        type = illegal_path.split('/')[-3]
        files_list = glob.glob(os.path.join(illegal_path, '*', '*'))
        for file in files_list:
            fname = os.path.basename(file)
            shutil.move(file, f'A:\\test01\\{tr_vl}\\{type}\\{fname}')
if __name__ == '__main__':
    # rename_legal()
    # rename_illegal()
    restruct_folder()