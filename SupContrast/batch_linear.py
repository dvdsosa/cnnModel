import os
import re
import subprocess

# Set the directory you want to start from
rootDir = '/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/path_models/SupCon_path_resnet50timm_lr_0.0125_decay_0.0001_bsz_32_temp_0.07_trial_0_cosine_warm'

# Function to extract the epoch number from the filename
def extract_epoch_number(filename):
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0

for dirName, subdirList, fileList in os.walk(rootDir):

    # Sort fileList based on the epoch number
    sorted_fileList = sorted(fileList, key=extract_epoch_number)
    
    for fname in sorted_fileList:
        # Generate the command for each file
        # command = f'python3 /home/dsosatr/tesis/cnnmodel/SupContrast/main_linear.py --batch_size 512 --learning_rate 0.25 --ckpt {os.path.join(dirName, fname)}'
        command = f'python3 /home/dsosatr/tesis/cnnmodel/SupContrast/main_linear.py --batch_size 512 --learning_rate 0.25 --ckpt {os.path.join(dirName, fname)} --mean "0.0361, 0.0326, 0.0357" --std "0.0997, 0.0968, 0.0858" --dataset path --data_folder /home/dsosatr/tesis/DYBtrainPadded/ --val_folder /home/dsosatr/tesis/DYBtestPadded/ --size 224'
        print(command)
        # Execute the command
        subprocess.run(command, shell=True, check=True)