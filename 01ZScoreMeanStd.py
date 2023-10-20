"""
Snippet of code to calculate the std and mean of the training dataset.

These values represent the mean and std of the training dataset for the Z-Score
formula, which will be used in main_supcon.py when supplying a custom dataset to 
the SupContrast model.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Define the MinMaxNormalization function
def min_max_normalization(image):
    return (image - image.min()) / (image.max() - image.min())

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(224),  # Rescale the shortest side of the image to 224
    transforms.Pad(2, fill=(255, 255, 255)),  # Add padding of size 2 to the image
    transforms.CenterCrop(224),  # Crop the center of the image to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    min_max_normalization,  # Apply min-max normalization
])

# Load the dataset
dataset = datasets.ImageFolder(root='/home/dsosatr/tesis/DYB-PlanktonNet', transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for the training and testing sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize lists to store the mean and std values
mean = torch.zeros(3)
std = torch.zeros(3)

print('==> Computing mean and std of the training dataset')
for inputs, _labels in tqdm(train_loader):
    for i in range(3):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()

# Final step
mean.div_(len(train_dataset))
std.div_(len(train_dataset))

print(f"Mean value: {mean}, \nStd value: {std}")

torch.save(train_dataset, '../train_dataset_DYB-PlanktonNet.pth')
torch.save(test_dataset, '../test_dataset_DYB-PlanktonNet.pth')

# Load the dataset
# train_dataset = torch.load('train_dataset_DYB-PlanktonNet.pth')
# test_dataset = torch.load('test_dataset_DYB-PlanktonNet.pth')

# results with full dataset: 
# tensor([0.0011, 0.0010, 0.0010])
# tensor([0.0022, 0.0022, 0.0019])

# Mean value: tensor([0.0011, 0.0010, 0.0010]),
# Std value: tensor([0.0023, 0.0022, 0.0019])