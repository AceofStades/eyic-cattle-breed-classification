import torch

DEVICE = torch.device("cuda:0")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

print(DEVICE)
