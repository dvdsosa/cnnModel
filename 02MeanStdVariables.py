"""
Snippet of code to compute the std and mean of the dataset.

These values represent the mean and std of the training dataset for the Z-Score
formula, which will be used in main_supcon.py when supplying a custom dataset to 
the SupContrast model.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Assuming your images are stored in 'PATH_TO_IMAGES'
PATH_TO_IMAGES = '/home/dsosatr/tesis/DYB-linearHead/train'
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder(root=PATH_TO_IMAGES, transform=transform)
loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

# Method 1
# Initialize lists to store the mean and std values
mean = torch.zeros(3)
std = torch.zeros(3)

print('==> Computing mean and std of the training dataset')
for inputs, _labels in tqdm(loader):
    for i in range(3):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()

# Final step
mean.div_(len(loader))
std.div_(len(loader))
print(f"Method 1:\n Mean value: {mean}, Std value: {std}")
# /home/dsosatr/tesis/DYBtrainCropped
#Method 1:
# Mean value: tensor([0.0613, 0.0559, 0.0583]), Std value: tensor([0.1215, 0.1185, 0.1019])

# Method 2
# For 3 channels images, adapted from https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/20
mean = torch.zeros(3)
meansq = torch.zeros(3)
count = 0

for data, _labels in tqdm(loader):
    for i in range(3):  # Assuming data is in the format (C, H, W)
        mean[i] += data[:, i, :, :].sum()
        meansq[i] += (data[:, i, :, :]**2).sum()
    count += np.prod(data.shape[2:])  # Only considering H and W for count

total_mean = mean/count
total_var = (meansq/count) - (total_mean**2)
total_std = torch.sqrt(total_var)

print(f'Method 2:\n Mean: {total_mean}, Std: {total_std}, Total pixels: {count}')
# Mean and Std values for DYB-cosine/train with 5 classes removed.
# Method 1:
# Mean value: tensor([0.0504, 0.0440, 0.0491]), Std value: tensor([0.1060, 0.1033, 0.0891])
# Method 2: used values for my thesis
# Mean: tensor([0.0419, 0.0355, 0.0410]), Std: tensor([0.0959, 0.0913, 0.0771]), Total pixels: 7588451567  --> ESTE ES EL UTILIZADO

# Mean and Std values for DYB-linearHead/train with 5 classes removed.
# Method 1:
#  Mean value: tensor([0.0503, 0.0439, 0.0491]), Std value: tensor([0.1060, 0.1033, 0.0891])
# Method 2:
#  Mean: tensor([0.0418, 0.0353, 0.0409]), Std: tensor([0.0956, 0.0911, 0.0769]), Total pixels: 6679209485   --> ESTE ES EL UTILIZADO