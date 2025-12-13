import torch

path = r"D:\MMI-MST-FORMER\1_data\processed\grape_test_images1.pt"
data = torch.load(path, weights_only=False)

print("Type:", type(data))
print("Shape:", data.shape)
print(data)
