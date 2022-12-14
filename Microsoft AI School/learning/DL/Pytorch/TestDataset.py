import glob
import os.path
from torch.utils.data.dataset import Dataset
import pandas as pd


class MyCustomDataset(Dataset):
    def __init__(self, path):
        # Data path <- 'C:\\Users\\user\\Desktop\\project\\file_csv'
        self.all_data_path = glob.glob(os.path.join(path, '*.csv'))
        csv_path = self.all_data_path[0]
        df = pd.read_csv(csv_path)
        self.filename = df.iloc[:, 1].values
        self.xywh = df.iloc[:, 2:].values

    def __getitem__(self, index):
        file_name = self.filename[index]
        xywh_ = self.xywh[index]

        return file_name, xywh_

    def __len__(self):
        return len(self.filename)


A = MyCustomDataset('C:\\Users\\user\\Desktop\\project\\file_csv')
for i in A:
    print(i)
