import os.path
import os
import glob
import torch
import copy
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torchvision.models as models
import torch.nn as nn


def visualize_aug(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))
    ])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))

    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def train(num_epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, device):
    print('start training...!')
    running_loss = 0
    total = 0
    best_loss = 9999
    for epoch in range(1, num_epoch + 1):
        for i, (imgs, labels) in enumerate(train_loader):
            img, label = imgs.to(device).float(), labels.to(device).long()
            output = model(img)

            loss = criterion(output, label)
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, argmax = torch.max(output, 1)
            acc = (label == argmax).float().mean()
            total += label.size(0)

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step[{}/{}] Loss : {:.4f} ACC : {:.2f}%'.format(
                    epoch + 1, num_epoch, i + 1, len(train_loader), loss.item(), acc.item() * 100
                ))

        avrg_loss, val_acc = validation(model, val_loader, criterion, device)

        if avrg_loss < best_loss:
            print(f'Best save at epoch >> {epoch}')
            print('save model in ', save_dir)
            best_loss = avrg_loss
            save_model(model, save_dir)

    save_model(model, save_dir, file_name='last_resnet.pt')


def validation(model, val_loader, criterion, device):
    print('start val...!')

    with torch.no_grad():
        model.eval()

        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device).float(), labels.to(device).long()
            output = model(imgs)
            loss = criterion(output, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(output, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print('val ACC : {:.2f}% avg_loss : {:.4f}'.format(
        val_acc, avrg_loss
    ))
    model.train()

    return avrg_loss, val_acc


def test_show(test_loader, device):
    net = models.__dict__['resnet50'](pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 53)
    net.to(device)

    model_path = '.\\weights\\best_resnet.pt'
    net.load_state_dict(torch.load(model_path))

    test_data_path = '.\\test'

    label_dict = folder_name_det(test_data_path)

    net.eval()
    with torch.no_grad():
        for i, (imgs, labels, path) in enumerate(test_loader):
            inputs, outputs, paths = imgs.to(device).float(), labels.to(device).long(), path
            import cv2
            img = cv2.imread(paths[0])
            predicted_outputs = net(inputs)
            _, predicted = torch.max(predicted_outputs, 1)

            labels_temp = labels.item()
            pr_temp = predicted.item()
            predicted_label = label_dict[str(pr_temp)]
            answer_label = label_dict[str(labels_temp)]
            cv2.putText(img, predicted_label, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(img, answer_label, (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.imshow('test', img)
            cv2.waitKey()

def save_model(model, save_dir, file_name='best_resnet.pt'):
    output_path = os.path.join(save_dir, file_name)

    torch.save(model.state_dict(), output_path)


def test_species(test_loader, device):
    net = models.__dict__['resnet50'](pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 53)
    net.to(device)

    model_path = '.\\weights\\best_resnet.pt'
    net.load_state_dict(torch.load(model_path))

    test_data_path = '.\\test'
    label_det = folder_name_det(test_data_path)
    label_length = len(label_det)
    labels_correct = list(0. for i in range(label_length))
    labels_total = list(0. for i in range(label_length))

    total = 0
    correct = 0
    net.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            inputs, outputs = imgs.to(device).float(), labels.to(device).long()
            predicted_outputs = net(inputs)
            _, predicted = torch.max(predicted_outputs, 1)

            labels_correct_running = (predicted == outputs).squeeze()
            label = outputs[0]
            if labels_correct_running.item():
                labels_correct[label] += 1
            labels_total[label] += 1
            total += inputs.size(0)
            correct += (outputs == predicted).sum().item()
        acc = correct / total * 100
    label_list = list(label_det.values())
    for i in range(53):
        # print('Accruacy to predict %5s : %2d % %'%(label_list[i], 100*labels_correct[i] / labels_total[i]))
        print("Accuracy to predict %5s : %2d %%" % (label_list[i], 100 * labels_correct[i] / labels_total[i]))
    print(f'Accuracy : {round(acc, 2)}%')


def folder_name_det(folder_path):
    folder_name = glob.glob(os.path.join(folder_path, "*"))
    det = {}
    for index, path in enumerate(folder_name):
        temp_name = path.split('\\')[-1]
        det[str(index)] = temp_name

    return det
