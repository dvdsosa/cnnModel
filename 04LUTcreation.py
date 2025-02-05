import torch
import os
import numpy as np
import sqlite3

import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from SupContrast.networks.resnet_big import FeatureExtractor
from SupContrast.networks.resnet_big import SupConResNet, LinearClassifier

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from torch.utils.data import DataLoader

# # Check if CUDA is available and set the device accordingly
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Instantiate the feature extractor
# feature_extractor = FeatureExtractor(name='resnet50timm')

# # Load the pre-trained weights
# state_load = torch.load('/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/path_models/SupCon_path_resnet50timm_lr_0.016_decay_0.0001_bsz_32_temp_0.07_trial_0_cosine_warm_falta/ckpt_epoch_850.pth')

# ###################################
# # Filter out the keys for the projection head
# #state_dict = {k: v for k, v in state_load['model'].items() if 'head' not in k}

# # This code is a more verbose way of the above line
# # Initialize an empty dictionary to store the filtered state dict
# filtered_state_dict = {}

# # Get the state dict from the loaded state
# loaded_state_dict = state_load['model']

# # Loop over each item in the loaded state dict
# for key, value in loaded_state_dict.items():
#     # Check if the key does not contain the string 'head'
#     if 'head' not in key:
#         # If the key does not contain 'head', add it to the filtered state dict
#         filtered_state_dict[key] = value
#     else:
#         print(f"Projection head found!: {key}")

# # Now, filtered_state_dict contains the same items as the loaded state dict,
# # but without the items where the key contains 'head'
# state_dict = filtered_state_dict
# ###################################

# # Load the pre-trained weights into the feature extractor
# feature_extractor.load_state_dict(state_dict, strict=False)

# # Set the model in evaluation mode. This is important to prevent the BatchNorm
# # and Dropout layers from changing their behavior
# # Tested and not difference in the "behaviour"
# #feature_extractor.eval()

# # Move the feature extractor to the GPU
# feature_extractor.to(device)

# # Define the transformation. 
# transformada = transforms.Compose([
#     transforms.ToTensor(), # Convert the image to a tensor
#     transforms.Normalize(mean=[0.0419, 0.0355, 0.0410], std=[0.0959, 0.0913, 0.0771]), # Perform the same normalization as the one used while training
# ])

# # Connect to the SQLite database
# conn = sqlite3.connect('trainingSetNew.db') # image_features.db
# # Create a cursor object
# c = conn.cursor()
# # Create table
# # https://sqlbolt.com/lesson/creating_tables
# c.execute('''CREATE TABLE IF NOT EXISTS features (
#           image_id INTEGER PRIMARY KEY,
#           class TEXT NOT NULL, 
#           image_name TEXT NOT NULL,
#           tensor BLOB NOT NULL
# )''')

# def get_file_info(root_folder, transformada):

#     train_dataset = datasets.ImageFolder(root=root_folder, transform=transformada)

#     # Create a data loader
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

#     for idx, (images, labels) in tqdm(enumerate(train_loader)):

#         # Display the first image in the pair
#         # plt.figure(figsize=(5, 5))
#         # plt.subplot(1, 2, 1)
#         # plt.imshow(images[0].permute(1, 2, 0))
#         # plt.title(train_dataset.classes[labels[0]])

#         # plt.show()

#         # Move the image to the GPU
#         images = images.to(device)

#         # Disables gradient calculation for the inference step. This can 
#         # be particularly useful when you’re working with large models or 
#         # large inputs, where memory usage could be a concern.
#         with torch.no_grad():
#             # Use the feature extractor
#             feature_vector = feature_extractor(images)

#         # Remove the dimension of size 1
#         feature_vector = feature_vector.squeeze(0)
#         # Convert tensor to bytes
#         buffer = BytesIO()
#         torch.save(feature_vector, buffer)
#         feature_vector_bytes = buffer.getvalue()

#         # You can get the filename from the path
#         full_path = train_dataset.samples[idx][0]
#         filename = os.path.basename(full_path)
#         class_name = train_dataset.classes[labels[0]]

#         # Insert a row of data via parameterized query to prevent SQL injection
#         c.execute("INSERT INTO features (class, image_name, tensor) VALUES (?, ?, ?)", (class_name, filename, feature_vector_bytes))

#         #print(f"Full path: {full_path}")
#         #print(f"File name: {filename}")
#         #print(f"Subfolder name: {class_name}")
#         #print("-------------------")

# # Replace 'root_folder' with the path to your root folder
# get_file_info('/home/dsosatr/tesis/DYBtrainCropped/', transformada)

# # Save (commit) the changes
# conn.commit()

# # Close the connection
# conn.close()

def store_features(val_loader, model):
    """validation"""
    model.eval()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            image_feature_vector = model.encoder(images)
            print("La ultima etiqueta es: ", labels[-1])

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion

def set_loader(opt):    
    mean = (0.0418, 0.0353, 0.0409)
    std = (0.0956, 0.0911, 0.0769)
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smallest side to 256
        transforms.CenterCrop(224),  # Crop the center of the image
        transforms.ToTensor(),  # Convert the image to a tensor
        normalize  # Normalize the image
    ])

    val_dataset = datasets.ImageFolder(root=opt.dataset_path,
                            transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader

class Options:
    def __init__(self):
        self.dataset_path = '/home/dsosatr/tesis/DYB-linearHead/train/'
        self.batch_size = 256
        self.model = 'resnet50timm'
        self.n_cls = 87
        self.ckpt = '/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/path_models/SupCon_path_resnet50timm_lr_0.016_decay_0.0001_bsz_32_temp_0.07_trial_0_cosine_warm/ckpt_epoch_900.pth'

def main():

    opt = Options()
    
    # build data loader
    val_loader = set_loader(opt)
    
    # build model and criterion
    model, classifier, criterion = set_model(opt)
    
    store_features(val_loader, model)
    
    pass


if __name__ == '__main__':
    main()
