import os.path

import torch


def train(num_epoch, model, train_loader, val_loader, criterion, optimizer, save_dir, device):
    print('Start Training ...')
    total = 0
    best_loss = 9999

    for epoch in range(1, num_epoch + 1):
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (argmax == label).float().mean()

            total += label.size(0)

            if (i + 1) % 10 == 0:
                print("Epoch >> [{} / {}],\t Step >> [{} / {}],\t Acc >> {:.2f}%".format(
                    epoch + 1, num_epoch, i + 1, len(train_loader), acc.item() * 100
                ))
        avg_loss, val_acc = validation(model, val_loader, criterion, device)
        if avg_loss < best_loss:
            print('Val Loss have been changed !!!')
            best_loss = avg_loss
            save_model(model, save_dir)


def save_model(model, save_dir, file_name='best.pt'):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)


def validation(model, val_loader, criterion, device):
    print('Val Start ... !')
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (img, label) in enumerate(val_loader):
            img, label = img.to(device), label.to(device)
            outputs = model(img)
            loss = criterion(outputs, label)
            batch_loss += loss.item()

            total += img.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (argmax == label).sum().item()
            total_loss += loss.item()
            cnt += 1

    avg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print('Acc >> {:.2f} Average Loss >> {:.4f}'.format(
        val_acc,
        avg_loss
    ))

    model.train()
    return avg_loss, val_acc
