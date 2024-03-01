import os

# Replace this with the actual path to your specific folder
specific_folder_path = '/home/dsosatr/Vitis-AI-3.0/DYB-Padded/train'

# Get a list of folder names in the specified directory
folder_names = [name for name in os.listdir(specific_folder_path) if os.path.isdir(os.path.join(specific_folder_path, name))]

# Sort the folder names alphabetically
folder_names.sort()

# Write the sorted folder names to a text file
with open('output.txt', 'w') as output_file:
    for folder_name in folder_names:
        output_file.write(f'"{folder_name}",\n')

print("Folder names have been written to 'output.txt'.")