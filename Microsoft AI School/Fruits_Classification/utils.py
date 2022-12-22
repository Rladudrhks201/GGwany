import torch
import os
import torch.nn as nn
from metric_monitor import MetricMonitor
from tqdm import tqdm


# Model Save
def save_model(model, save_dir, file_name='last.pt'):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print('mult gpu 저장 !')
        torch.save(model.module.state_dict(), output_path)
    else:
        print('single gpu 저장 !')
        torch.save(model.state_dict(), output_path)


# Train, Val Loop
def train(number_epoch, train_loader, val_loader, criterion, optimizer, model, save_dir, device):
    print('start training...')
    running_loss = 0.0
    total = 0
    best_loss = 91919

    for epoch in range(number_epoch):
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, argmax = torch.max(outputs, 1)
            acc = (labels == argmax).float().mean()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(
                    epoch + 1, number_epoch, i + 1, len(train_loader), loss.item(), acc.item() * 100
                ))

        avg_loss, val_acc = validate(epoch, model, val_loader, criterion, device)

        # 특정 epoch 마다 저장 하고 싶다 하는 경우
        if epoch % 10 == 0:
            save_model(model, save_dir, file_name=f'{epoch}.pt')
        # best save
        if val_acc > best_loss:
            print(f'best save epoch ! >>> {epoch}')
            best_loss = val_acc
            save_model(model, save_dir, file_name='best.pt')

    save_model(model, save_dir, file_name='final.pt')

def validate(epoch, model, val_loader, criterion, device):
    print('start validation...')
    with torch.no_grad():  # 학습 X
        model.eval()
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (images, labels) in tqdm(enumerate(val_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += labels.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (argmax == labels).sum().item()
            total_loss += loss.item()
            cnt += 1

    avg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("val # {} acc {:.2f}$ avg loss {:.4f}".format(
        epoch + 1, val_acc, avg_loss
    ))
    model.train()   # val 에서 다시 train 단계로
    return avg_loss, val_acc