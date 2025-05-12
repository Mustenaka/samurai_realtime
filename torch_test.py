import torch

available = torch.cuda.is_available()
print(available)

backend = torch.backends.cudnn.is_available()
print(backend)