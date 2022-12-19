from DataSet import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train aug
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])
# val aug
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

# dataset
train_dataset = custom_dataset('C:\\Users\\user\\Desktop\\Search\\Data\\train', transform=train_transform)
val_dataset = custom_dataset('C:\\Users\\user\\Desktop\\Search\\Data\\val', transform=val_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
