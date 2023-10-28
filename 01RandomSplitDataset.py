"""
Snippet of code to create the training and test sets.

We use 2 type of transformations:
- transform1: first rescaling, then cropping at the center. Parts of the image will be lost.
The `CenterCrop` operation is often used when you want to crop the 
center part of an image after resizing, especially when the aspect 
ratio changes during resizing. It helps to focus on the center of 
the image where the main subject often resides. 

- transform2: first padding, then rescaling. The image will be square with added black pixels.
Here the images are transformed using a padding to make them square, then resized
to 224x224 and finally normalized using min-max normalization.

If your images' main subjects are not centrally located, cropping might 
remove important information. On the other hand, if your images 
have a lot of background noise around the edges, cropping might be 
beneficial. Always consider these factors when designing your 
preprocessing pipeline.

In general, it is a good idea to experiment with different options and 
see what works best for your specific task. You could train models with 
both types of padding and compare their performance on a validation set 
to decide which approach is better. Remember that what works best can 
vary depending on the specific task and dataset.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Function to save images
def save_images(dataset, folder):
    for i, (image, label) in tqdm(enumerate(dataset), desc='Saving images'):
        class_name = dataset.dataset.classes[label]
        file_name = os.path.basename(dataset.dataset.imgs[dataset.indices[i]][0])
        class_folder = os.path.join(folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        pil_image = TF.to_pil_image(image)
        pil_image.save(os.path.join(class_folder, f'{file_name}'))

# The padding is evenly distributed on both sides of the image, making the image square
def pad_image(img):
    # Get size
    w, h = img.size
    max_wh = max(w, h)
    hp = (max_wh - w) // 2
    vp = (max_wh - h) // 2
    # Adjust padding for odd differences
    hp_odd = (max_wh - w) % 2
    vp_odd = (max_wh - h) % 2
    padded_img = transforms.functional.pad(img, (hp, vp, hp + hp_odd, vp + vp_odd), fill=(0, 0, 0))
    # print(f'After padding, the image size is: {padded_img.size}')
    return padded_img

# Define the MinMaxScaling function
# Not needed because this is done by ToTensor() (which would normalize the data to [0, 1]).
def min_max_scaling(image):
    return (image - image.min()) / (image.max() - image.min())

# Define the transformation, first rescaling, then cropping at the center. Parts of the image will be lost.
transform1 = transforms.Compose([
    transforms.Resize(224), # Rescale the shortest side of the image to 224
    transforms.CenterCrop(224), # Crop a square in the center of the image
    transforms.ToTensor(), # Convert the image to a tensor
])

# Define the transformation
transform2 = transforms.Compose([
    transforms.Lambda(lambda img: pad_image(img)),
    transforms.Resize(224),  # Rescale the image to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
])

# Load the dataset
dataset1 = datasets.ImageFolder(root='/home/dsosatr/tesis/DYB-PlanktonNet', transform=transform1)
dataset2 = datasets.ImageFolder(root='/home/dsosatr/tesis/DYB-PlanktonNet', transform=transform2)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset1))  # 80% for training
test_size = len(dataset1) - train_size  # 20% for testing

train_dataset1, test_dataset1 = random_split(dataset1, [train_size, test_size])
train_dataset2, test_dataset2 = random_split(dataset2, [train_size, test_size])

# Create a subset of the training and testing sets
train_subset1 = Subset(dataset1, train_dataset1.indices)
test_subset1 = Subset(dataset1, test_dataset1.indices)

train_subset2 = Subset(dataset2, train_dataset2.indices)
test_subset2 = Subset(dataset2, test_dataset2.indices)

# Save images from train_subset and test_subset in a new folder
save_images(train_subset1, '/home/dsosatr/tesis/DYBtrainCropped')
save_images(test_subset1, '/home/dsosatr/tesis/DYBtestCropped')

# Save images from train_subset and test_subset in a new folder
save_images(train_subset2, '/home/dsosatr/tesis/DYBtrainPadded')
save_images(test_subset2, '/home/dsosatr/tesis/DYBtestPadded')
