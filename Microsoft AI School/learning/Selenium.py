import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import cv2
import pandas as pd
import os
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.model_selection import train_test_split


# options = webdriver.ChromeOptions()
# options.add_experimental_option("detach", True)
# driver = webdriver.Chrome(options=options)
# driver.implicitly_wait(10)
# driver.get('https://www.google.com')
#
# elem = driver.find_element(By.NAME, 'q')
# elem.clear()
# elem.send_keys('banana')
# elem.send_keys(Keys.RETURN)  # ENTER 입력
# assert "No results found." not in driver.page_source
#
#
#
# # 이미지 메뉴 누르기
# driver.find_element(By.XPATH, '/html/body/div[7]/div/div[4]/div/div[1]/div/div[1]/div/div[2]/a').click()
# selenium_scroll_option(driver)
#
# driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input').click()
# selenium_scroll_option(driver)
# img_srcs = driver.find_element(By.CLASS_NAME, 'rg_i')


# main()
drink = ['Coke', 'Sprite', 'Milkis']

for i in drink:
    for j in range(41):
        img = cv2.imread(f"C:/Users/user/Documents/github/Microsoft AI School/learning/data/{i}/{j}.png")
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        img2 = cv2.resize(img2, (28, 28))
        cv2.imwrite(f"C:/Users/user/Documents/github/Microsoft AI School/learning/data/{i}/{j}.png", img2)


Coke = {
    'file_name': [f'{i}.png' for i in range(41)]
}
Sprite = {
    'file_name': [f'{i}.png' for i in range(41)]
}
Milkis = {
    'file_name': [f'{i}.png' for i in range(41)]
}

coke = pd.DataFrame(Coke, columns=['file_name', 'label'])
sprite = pd.DataFrame(Sprite, columns=['file_name', 'label'])
milkis = pd.DataFrame(Milkis, columns=['file_name', 'label'])
coke['label'] = 0
sprite['label'] = 1
milkis['label'] = 2

df = pd.concat([coke, sprite, milkis], axis=0)
df.to_csv('./data/beverage.csv')

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'], skiprows=[0])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        try:
            image = read_image(img_path)
        except:
            print(self.img_labels.iloc[idx, 0])
            exit()
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# linear Neural Networks Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784)))  # 784 = 28*28, Flatten과 같은 역할
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return F.log_softmax(h6, dim=1)


# 설정

epochs = 10
lr = 0.01
momentum = 0.5  # 관성, 경사 하강법에서 최저점을 찾을때, 더 나아가서 더 낮은 점이 있는지 확인할 때 사용
no_cuda = True
seed = 1
log_interval = 200  # 시스템 로그

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device('cuda' if use_cuda else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# 데이터 셋 로드

test_batch_size = 64  # 1000

os.makedirs('C:/Users/user/Desktop/data/bevtr', exist_ok=True)
os.makedirs('C:/Users/user/Desktop/data/bevte', exist_ok=True)

dt1 = pd.read_csv('./data/beverage.csv')
X_tr, X_te, Y_tr, Y_te = train_test_split(dt1.iloc[:, 1], dt1.iloc[:, 2], test_size=0.17)
tr1 = pd.concat([X_tr, Y_tr], axis=1)
te1 = pd.concat([X_te, Y_te], axis=1)
tr1.to_csv('./data/bevtr.csv')
te1.to_csv('./data/bevte.csv')
for i, j in tr1['label'], tr1['file_name']:
    if i == 0:
        img = cv2.imread(f"C:/Users/user/Documents/github/Microsoft AI School/learning/data/Coke/{j}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('C:/Users/user/Desktop/data/bevtr/', img)
    elif i ==1:
        img = cv2.imread(f"C:/Users/user/Documents/github/Microsoft AI School/learning/data/Sprite/{j}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('C:/Users/user/Desktop/data/bevtr/', img)
    else:
        img = cv2.imread(f"C:/Users/user/Documents/github/Microsoft AI School/learning/data/Milkis/{j}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('C:/Users/user/Desktop/data/bevtr/', img)


trainset = CustomImageDataset(
    annotations_file='./data/bevtr.csv',
    img_dir='C:/Users/user/Desktop/data/bevtr/'
)
testset = CustomImageDataset(
    annotations_file='./data/bevte.csv',
    img_dir='C:/Users/user/Desktop/data/bevte/'
)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=test_batch_size, shuffle=True, **kwargs)

# 훈련

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # data -> 형태 : 64 (batch size), 1, 28, 28
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(log_interval, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))


for epoch in range(1, 11):
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)