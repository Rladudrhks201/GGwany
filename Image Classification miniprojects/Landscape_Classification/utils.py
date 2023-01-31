import matplotlib.pyplot as plt
import numpy as np
import torch


def train(n_epochs, train_loader, val_loader, model, optimizer, criterion, device, save_path,
          last_validation_loss=None):
    valid_loss_min = np.Inf
    train_loss_ls = []
    valid_loss_ls = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        trainLoss = 0.0
        train_batch = 0
        validLoss = 0.0
        valid_batch = 0
        model.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            datas, targets = imgs.to(device), labels.to(device)
            output = model(datas)

            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update train_loss
            train_loss += ((1 / (batch_idx + 1))) * (loss.data - train_loss)
            trainLoss += train_loss
            train_batch += 1

        # val model
        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            datas, targets = data.to(device), target.to(device)
            output = model(datas)
            loss = criterion(output, targets)

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            validLoss += valid_loss.item()
            valid_batch += 1

        print('Epoch {} \t Training Loss : {:.6f} \t Validation Loss : {:.6f}'.format(
            epoch, train_loss, valid_loss
        ))

        train_loss_ls.append(trainLoss / train_batch)
        valid_loss_ls.append(validLoss / valid_batch)

        if valid_loss <= valid_loss_min:
            print('Validation Loss Decreased ({:.6f} -- > {:.6f}.) Saving Model ...'.format(
                valid_loss_min, valid_loss
            ))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # Visualize loss graph
    plt.plot(train_loss_ls, '-o')
    plt.plot(valid_loss_ls, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Val'])
    plt.show()

def acc_function(correct, total):
    acc = correct / total * 100
    return acc

def test(model, data_loader, device):
    model_path = '.\\model\\best.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            images, labels = image.to(device), label.to(device)
            output = model(images)
            _, argmax = torch.max(output, 1)
            total += images.size(0)
            correct += (labels == argmax).sum().item()

        acc = acc_function(correct, total)
        print(f'acc >> {acc}%')

