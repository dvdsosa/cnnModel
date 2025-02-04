#!/bin/bash

# Define the directory containing the files
dir="/home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/path_models/SupCon_path_resnet50timm_lr_0.016_decay_0.0001_bsz_32_temp_0.07_trial_0_cosine_warm"

# Loop over each .pth file in the directory in reverse order
ls -r "$dir"/*.pth | sort -V -r | while read -r file
do
  echo "Working on file $file"
  # Construct the command
  cmd="python main_linear.py --batch_size 256 --learning_rate 2.5 --model resnet50timm --ckpt \"$file\" --mean \"0.0418, 0.0353, 0.0409\" --std \"0.0956, 0.0911, 0.0769\" --dataset path --data_folder /home/dsosatr/tesis/DYB-linearHead/train/ --val_folder /home/dsosatr/tesis/DYB-linearHead/val/ --size 224"

  #echo $cmd
  # Execute the command
  eval $cmd
done
