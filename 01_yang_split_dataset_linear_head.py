import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    """
    Split plankton image dataset into training, validation, and testing sets.
    
    This function processes images from a source directory containing subdirectories for
    each plankton class and splits them into training (70%), validation (15%), and testing (15%) sets.
    The split is performed class-wise, maintaining the same class distribution across all sets.
    Files are copied from the source directory to their respective destination directories.
    
    The split is done in two steps:
    1. First split: 70% training, 30% temporary
    2. Second split: The temporary set is divided equally into validation and test sets
    
    Source and destination directories are defined as constants within the function.
    """
    # Define your source and destination directories
    source_dir = '/home/dsosatr/tesis/DYB-PlanktonNet'
    train_dir = '/home/dsosatr/tesis/DYB-linearHead/train'
    val_dir = '/home/dsosatr/tesis/DYB-linearHead/val'
    test_dir = '/home/dsosatr/tesis/DYB-linearHead/test'

    # Get the list of subdirectories (classes)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(source_dir, class_name)
        
        # Get the list of files in the class directory
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Split the files into training, validation, and testing sets
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        # Create corresponding directories in the destination directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copy the files into the destination directories
        for file in train_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(train_class_dir, file))
        for file in val_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(val_class_dir, file))
        for file in test_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(test_class_dir, file))


if __name__ == "__main__":
    main()
