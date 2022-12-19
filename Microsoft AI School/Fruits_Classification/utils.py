import torch
import os
import torch.nn as nn
# train Loop

# Val loop

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


# Model Evaluate
def calculate_acc(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target ==1.0
    return torch.true_divide((output == target).sum(dim=0), output.size(0)).item()

