import torch
print(torch.cuda.is_available())  # Should output True
print(torch.cuda.get_device_name(0))  # Should show "NVIDIA GeForce RTX 3050"
