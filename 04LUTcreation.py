import torch
import os
import numpy as np
import sqlite3
import faiss

import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from SupContrast.networks.resnet_big import FeatureExtractor
from SupContrast.networks.resnet_big import SupConResNet, LinearClassifier
from tqdm import tqdm

def init_db(conn, cursor):
    """
    Initializes the database by creating the necessary table for FAISS ID to label mapping.

    Args:
        conn (sqlite3.Connection): The connection object to the SQLite database.
        cursor (sqlite3.Cursor): The cursor object to execute SQL commands.

    Creates:
        - feature_mappings: Maps FAISS IDs to labels
    """
    # Create the feature_mappings table for FAISS integration
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feature_mappings (
        faiss_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL
    )
    ''')
    conn.commit()

def load_or_init_faiss_index(dim=2048):
    """
    Loads an existing FAISS index or initializes a new one if it doesn't exist.
    
    Args:
        dim (int): Dimension of feature vectors, default is 2048.
        
    Returns:
        faiss.IndexFlatIP: FAISS index for efficient similarity search.
    """
    index_path = 'faiss_index.bin'
    
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print(f"Loaded FAISS index with {index.ntotal} vectors")
            return index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
    
    # Create a new index if file doesn't exist or loading failed
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    print(f"Created new FAISS index with dimension {dim}")
    return index

def save_faiss_index(index):
    """
    Saves the FAISS index to disk.
    
    Args:
        index (faiss.Index): The FAISS index to save.
    """
    index_path = 'faiss_index.bin'
    try:
        faiss.write_index(index, index_path)
        print(f"FAISS index saved with {index.ntotal} vectors")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def process_image_batch(train_loader, model):
    """
    Processes a batch of images, extracts feature vectors using a given model, 
    and stores the features in a FAISS index and their mappings in an SQLite database.
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of images and labels.
        model (torch.nn.Module): PyTorch model with an encoder for feature extraction.
    Workflow:
        1. Sets the model to evaluation mode.
        2. Connects to an SQLite database and initializes it if necessary.
        3. Loads or initializes a FAISS index for storing feature vectors.
        4. Iterates over batches of images and labels from the DataLoader:
            - Extracts feature vectors using the model's encoder.
            - Normalizes the feature vectors for cosine similarity.
            - Adds the feature vectors to the FAISS index.
            - Maps the FAISS index IDs to their corresponding labels in the SQLite database.
        5. Commits changes to the database after processing each batch.
        6. Saves the FAISS index and closes the database connection.
    Notes:
        - The FAISS index is used for efficient similarity search.
        - The SQLite database stores mappings between FAISS IDs and labels.
        - Feature vectors are normalized to ensure compatibility with cosine similarity.
    Exceptions:
        - Catches and prints any exceptions that occur during batch processing.
    Outputs:
        - Prints the total number of vectors in the FAISS index and the number of vectors added.
    """

    model.eval()
    
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('plankton_db.sqlite')
    cursor = conn.cursor()
    init_db(conn, cursor)
    
    # Load or initialize FAISS index
    faiss_index = load_or_init_faiss_index(dim=2048)
    initial_index_size = faiss_index.ntotal
    
    try:
        with torch.no_grad():
            for idx, (images, labels) in enumerate(tqdm(train_loader)):
                images = images.float().cuda()
                bsz = len(labels)
                
                # Extract features
                image_feature_vector = model.encoder(images)
                
                # Process each feature vector
                for i, (label, feature_vector) in enumerate(zip(labels, image_feature_vector)):
                    # Normalize for cosine similarity (not performed on the model.encoder)
                    feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=0)
                    # FAISS requires Numpy arrays to be of type float32, stored in CPU memory, and flattened into a 1D structure for optimal processing.
                    feature_np = feature_vector.cpu().numpy().flatten().astype(np.float32)
                    
                    # Add vector to FAISS index
                    faiss_index.add(feature_np.reshape(1, -1))
                    
                    # Calculate the FAISS ID (index is zero-based)
                    faiss_id = initial_index_size + i + sum([len(train_loader.dataset) for _ in range(idx)])
                    
                    # Store mapping in SQLite
                    cursor.execute('''
                    INSERT INTO feature_mappings (faiss_id, label)
                    VALUES (?, ?)
                    ''', (int(faiss_id), label))
                    
                # Commit every batch to avoid data loss
                conn.commit()
    
    except Exception as e:
        print(f"Error processing batch: {e}")
    finally:
        # Save FAISS index and close database connection
        save_faiss_index(faiss_index)
        conn.close()
        print(f"FAISS index now contains {faiss_index.ntotal} vectors (added {faiss_index.ntotal - initial_index_size})")

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
        self.learning_rate = 0.001

def main():
    """
    Main function to execute the feature extraction pipeline.
    This function performs the following steps:
    1. Initializes options and configurations.
    2. Builds the data loader for the training dataset.
    3. Constructs the model, classifier, and loss criterion.
    4. Processes batches of images from the training dataset and stores
       their extracted features in a database (SQLite named plankton_db.sqlite) 
       and a FAISS index (named faiss_index.bin).
    Prints a confirmation message upon successful completion of the process.
    """

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
