import os
import csv
import matplotlib.pyplot as plt


def main():
    """
    Count and visualize the number of instances per class in a dataset.

    This script analyzes an image dataset organized in a hierarchical folder structure
    (train, val, test) and counts the number of images in each class folder. It then:
    1. Generates a bar chart showing the distribution of instances across classes
    2. Saves the distribution data to a CSV file named 'plankton_data.csv'
    3. Displays the visualization

    The dataset is expected to be organized as follows:
    base_path/
        train/
            class_1/
                image1.jpg
                ...
            class_2/
                ...
        val/
            ...
        test/
            ...
    """
    # Define the paths to the train, val, and test folders
    base_path = '/home/dsosatr/tesis/DYB-linearHead'
    subfolders = ['train', 'val', 'test']

    # Initialize a dictionary to hold the counts for each class
    class_counts = {}

    # Iterate over each subfolder (train, val, test)
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_path, subfolder)
        # Iterate over each class folder within the subfolder
        for class_folder in os.listdir(subfolder_path):
            class_folder_path = os.path.join(subfolder_path, class_folder)
            if os.path.isdir(class_folder_path):
                # Count the number of instances in the class folder
                num_instances = len(os.listdir(class_folder_path))
                # Add the count to the total for this class
                if class_folder not in class_counts:
                    class_counts[class_folder] = 0
                class_counts[class_folder] += num_instances

    # Sort the classes by their names
    sorted_classes = sorted(class_counts.keys())

    # Create a list of class labels (1, 2, 3, ...)
    class_labels = list(range(1, len(sorted_classes) + 1))

    # Create a list of instance counts in the same order as the class labels
    instance_counts = [class_counts[class_name] for class_name in sorted_classes]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, instance_counts, tick_label=sorted_classes)
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.title('Number of Instances per Class')

    # Set the x-ticks to show every 10 labels
    plt.xticks(ticks=class_labels[::10], labels=class_labels[::10])

    # Define the path to the output CSV file
    output_csv_path = '/home/dsosatr/tesis/cnnmodel/plankton_data.csv'

    # Write the class labels and instance counts to the CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Class', 'Instances'])
        # Write the data rows
        for class_name, instance_count in zip(class_labels, instance_counts):
            writer.writerow([class_name, instance_count])

    plt.show()

if __name__ == "__main__":
    main()
