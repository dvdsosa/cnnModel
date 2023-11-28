import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from SupContrast.networks.resnet_big import FeatureExtractor


########## STEP 1 ##########
# LOAD CIFAR10 DATASET

mean1 = (0.4914, 0.4822, 0.4465)
std1 = (0.2023, 0.1994, 0.2010)

normalize = transforms.Normalize(mean=mean1, std=std1)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    # The parameters (0.4, 0.4, 0.4, 0.1) correspond to the amount of 
    # jitter in brightness, contrast, saturation, and hue, respectively.
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])

train_dataset = datasets.CIFAR10(root="/home/dsosatr/tesis/cnnmodel/SupContrast/datasets/",
                                    transform=TwoCropTransform(train_transform),
                                    download=True)

train_sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=(train_sampler is None),
    num_workers=1, pin_memory=True, sampler=train_sampler)

########## STEP 2 ##########
# LOAD FEATURE EXTRACTOR

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the feature extractor
feature_extractor = FeatureExtractor(name='resnet50')

# Load the pre-trained weights
state_load = torch.load('/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.2_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/ckpt_epoch_1000.pth')

######
# Filter out the keys for the projection head (keep only the backbone for feature extraction)
#state_dict = {k: v for k, v in state_load['model'].items() if 'head' not in k}

# This code is a more verbose way of the above line
# Initialize an empty dictionary to store the filtered state dict
filtered_state_dict = {}

# Get the state dict from the loaded state
loaded_state_dict = state_load['model']

# Loop over each item in the loaded state dict
for key, value in loaded_state_dict.items():
    # Check if the key does not contain the string 'head'
    if 'head' not in key:
        # If the key does not contain 'head', add it to the filtered state dict
        filtered_state_dict[key] = value
    else:
        print(f"Projection head found!: {key}")

# Now, filtered_state_dict contains the same items as the loaded state dict,
# but without the items where the key contains 'head'
state_dict = filtered_state_dict
######

# Load the pre-trained weights into the feature extractor
feature_extractor.load_state_dict(state_dict, strict=False)

# Set the model in evaluation mode. This is important to prevent the BatchNorm
# and Dropout layers from changing their behavior
# Tested and not difference in the "behaviour"
#feature_extractor.eval()

# Move the feature extractor to the GPU
feature_extractor.to(device)

########## STEP 3 ##########
# GENERATE FEATURE VECTOR FOR EACH IMAGE AND STORE IT IN A LIST

# Initialize empty lists
classes_list = []
features_list = []

for idx, (images, labels) in tqdm(enumerate(train_loader)):

    # Break the loop after 1000 iterations
    if idx >= 1000:
        break

    # Display the first image in the pair
    # plt.figure(figsize=(5, 5))
    # plt.subplot(1, 2, 1)
    # Remove the dimension of size 1
    # image = images[0].squeeze(0)
    # plt.imshow(image.permute(1, 2, 0))
    # plt.title(train_loader.dataset.classes[labels[0]])
    # plt.show()

    image = images[0].to(device)

    with torch.no_grad():
        # Use the feature extractor
        feature_vector = feature_extractor(image)

    # Remove the dimension of size 1
    feature_vector = feature_vector.squeeze(0)

    features_list.append(feature_vector)
    classes_list.append(train_loader.dataset.classes[labels[0]])


########## STEP 4 ##########
# CALCULATE COSINE SIMILARITY

# Convert the list of tensors into a single tensor
tensors = torch.stack(features_list)

# Create a CosineSimilarity object
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# Calculate the cosine similarity between the first tensor and the rest
similarity = cos(tensors[0].unsqueeze(0), tensors)

# The result is a tensor of size 1000, where each element is the cosine similarity
# the maximum value correspond to the first image, which is the same image, so it 
# is working correctly.
