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

# Assuming your images are stored in 'path_to_images'
path_to_images = '/home/dsosatr/tesis/DYBtrainCropped'
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder(root=path_to_images, transform=transform)
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
# /home/dsosatr/tesis/DYBtrainCropped
# Method 2:
# Mean: tensor([0.0613, 0.0559, 0.0583]), Std: tensor([0.1330, 0.1289, 0.1111]), Total pixels: 1903426560

# Mean and Std values for DYBtrainCropped with 5 classes removed.
# Method 1:
#  Mean value: tensor([0.0613, 0.0559, 0.0583]), Std value: tensor([0.1217, 0.1186, 0.1019])
# Method 2:
#  Mean: tensor([0.0613, 0.0559, 0.0583]), Std: tensor([0.1332, 0.1291, 0.1111]), Total pixels: 1902121984
