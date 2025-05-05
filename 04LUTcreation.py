import torch
import os
import numpy as np
import sqlite3

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from SupContrast.networks.resnet_big import FeatureExtractor
from SupContrast.networks.resnet_big import SupConResNet, LinearClassifier
from tqdm import tqdm

def store_feature(label, feature_vector, cursor, conn):
    """
    Stores a feature vector in the database with the associated label.

    Args:
        label (str): The label associated with the feature vector.
        feature_vector (numpy.ndarray): The feature vector to be stored.
        cursor (sqlite3.Cursor): The database cursor for executing SQL commands.
        conn (sqlite3.Connection): The database connection object.

    Returns:
        None
    """
    cursor.execute('''
    INSERT INTO plankton_features (label, vector)
    VALUES (?, ?)
    ''', (label, feature_vector.tobytes()))
    conn.commit()

def init_db(conn, cursor):
    """
    Initializes the database by creating the 'plankton_features' table if it does not already exist.

    Args:
        conn (sqlite3.Connection): The connection object to the SQLite database.
        cursor (sqlite3.Cursor): The cursor object to execute SQL commands.

    The 'plankton_features' table has the following columns:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - label: TEXT NOT NULL
        - vector: BLOB NOT NULL

    Commits the transaction after executing the SQL command.
    """
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plankton_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        vector BLOB NOT NULL
    )
    ''')
    conn.commit()

def process_image_batch(train_loader, model):
    """
    Processes a batch of images using a given model and stores the resulting feature vectors in an SQLite database.
    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader providing batches of images and labels.
        model (torch.nn.Module): The neural network model used to extract feature vectors from images.
    Returns:
        None
    """
    model.eval()
    
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('plankton_db.sqlite')
    cursor = conn.cursor()
    init_db(conn, cursor)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.float().cuda()
            labels = labels
            bsz = len(labels)

            # forward
            image_feature_vector = model.encoder(images)
            for label, feature_vector in zip(labels, image_feature_vector):
                store_feature(label, feature_vector.cpu().numpy().flatten(), cursor, conn)

    conn.close()

def set_model(opt):
    """
    Initializes and sets up the model, classifier, and criterion for training or evaluation.

    Args:
        opt (Namespace): A namespace object containing the following attributes:
            - model (str): The name of the model to be used.
            - n_cls (int): The number of classes for the classifier.
            - ckpt (str): The path to the checkpoint file.

    Returns:
        tuple: A tuple containing the following elements:
            - model (torch.nn.Module): The initialized and loaded model.
            - classifier (torch.nn.Module): The initialized classifier.
            - criterion (torch.nn.Module): The loss function criterion.

    Raises:
        NotImplementedError: If no GPU is available.
    """
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

class CustomDataset(datasets.ImageFolder):
    """
    A custom dataset class that extends the ImageFolder dataset from torchvision.

    This class overrides the __getitem__ method to return the image tensor along with the class name
    instead of the class index.

    Methods
    -------
    __getitem__(index)
        Returns the image tensor and the class name for the given index.

    Parameters
    ----------
    index : int
        Index of the sample to be fetched.

    Returns
    -------
    tuple
        A tuple containing the image tensor and the class name.
    """
    def __getitem__(self, index):
        original_tuple = super(CustomDataset, self).__getitem__(index)
        path, _ = self.samples[index]
        class_name = self.classes[original_tuple[1]]
        return original_tuple[0], class_name

def set_loader(opt):
    """
    Sets up the data loader for the validation dataset.

    Args:
        opt (Namespace): A namespace object containing the following attributes:
            - dataset_path (str): Path to the dataset directory.
            - batch_size (int): Number of samples per batch to load.

    Returns:
        DataLoader: A PyTorch DataLoader for the validation dataset.
    """
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

    custom_val_dataset = CustomDataset(root=opt.dataset_path, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        custom_val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader

class Options:
    """
    A class to represent configuration options for a CNN model.

    Attributes
    ----------
    dataset_path : str
        Path to the dataset used for training.
    batch_size : int
        Number of samples per batch.
    model : str
        Name of the model architecture to be used.
    n_cls : int
        Number of classes in the dataset.
    ckpt : str
        Path to the checkpoint file for the model.

    Methods
    -------
    __init__():
        Initializes the Options class with default values.
    """
    def __init__(self):
        self.dataset_path = '/home/dsosatr/tesis/DYB-linearHead/train/'
        self.batch_size = 256
        self.model = 'resnet50timm'
        self.n_cls = 87
        self.ckpt = '/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/path_models/SupCon_path_resnet50timm_lr_0.016_decay_0.0001_bsz_32_temp_0.07_trial_0_cosine_warm/ckpt_epoch_900.pth'

def main():

    opt = Options()
    
    # build data loader
    train_loader = set_loader(opt)
    
    # build model and criterion
    model, classifier, criterion = set_model(opt)
    
    # process images and store features in database
    process_image_batch(train_loader, model)
    
    print("Finished saving features of training set to database")


if __name__ == '__main__':
    main()
