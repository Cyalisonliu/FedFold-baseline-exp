import torch
import torchvision

# CIFAR10
data_tensor = torch.zeros(500, 32, 32, 3).cuda()
print(data_tensor.shape)
memory_allocated = torch.cuda.memory_allocated()
print(f'GPU memory allocated: {memory_allocated/1024/1024} MB')


