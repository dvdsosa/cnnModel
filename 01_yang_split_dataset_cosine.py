import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    """
    Split plankton image dataset into training and testing sets.
    
    This function takes images from a source directory containing subdirectories for
    each plankton class and splits them into training (80%) and testing (20%) sets.
    The split is performed class-wise, maintaining the same class distribution in both sets.
    Files are copied from the source directory to their respective train and test directories.
    
    Source and destination directories are defined as constants within the function.
    """
    # Define your source and destination directories
    source_dir = '/home/dsosatr/tesis/DYB-PlanktonNet'
    train_dir = '/home/dsosatr/tesis/DYB-original/train'
    test_dir = '/home/dsosatr/tesis/DYB-original/test'

    # Get the list of subdirectories (classes)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(source_dir, class_name)
        
        # Get the list of files in the class directory
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Split the files into training and testing sets
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        
        # Create corresponding directories in the destination directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copy the files into the destination directories
        for file in train_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(train_class_dir, file))
        for file in test_files:
            shutil.copy(os.path.join(class_dir, file), os.path.join(test_class_dir, file))


if __name__ == "__main__":
    main()