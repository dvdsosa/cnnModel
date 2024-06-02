# SEResNeXt trained with SCL

## TASKS

### DONE
- [x] Preprocessing: create training and testing sets, 01RandomSplitDataset.py file.
- [x] Preprocessing: calculate mean and std for Z-Score formula, 02MeanStdVariables.py file.
- [x] Added Supervised Contrastive Learning (SCL) repository.
- [x] Added notification by email when it ends or breaks.
- [x] Added DYB-PlanktonNet dataset to main_supcon.py.
- [x] Added SeResNext50 model to SCL repository.
	
### TODO next day

- [] Perform LUT prunning.
- [x] Learn how to do the inference with this method.
- [] Calculate performance metrics Acc (top-1), Recall, Precision, F1-Score.
- [] Peform training with original size of 224 by 224.

## CNN Model

### Model training

- [x] *1$: train the CIFAR10 dataset with resnet50, --batch_size=128, --lr=0.2, --temp=0.07.
best accuracy: 94.88% for last epoch, trained for 1000 epochs, linear bs=256
- [x] *2$: train the CIFAR10 dataset with resnet50, --batch_size=128, --lr=0.2, --temp=0.1.
best accuracy: 95.02% for last epoch, trained for 1000 epochs, linear bs=256 lr=2.5
- [x] *3$: train the CIFAR10 dataset with resnet50, --batch_size=128, --lr=0.5, --temp=0.07.
best accuracy: 94.95% for epoch 1000, trained for 1000 epochs, linear bs=256
- [x] *4$: train the CIFAR10 dataset with resnet50, --batch_size=128, --lr=0.5, --temp=0.1.
best accuracy: 94.79% for last epoch, trained for 1000 epochs, linear bs=256
PAPER REFERENCE
- [x] *5$: train the CIFAR10 dataset with resnet50, --batch_size=512, --lr=0.25, --temp=0.1.
best accuracy: 94.52% for epoch 1000, trained for 1000 epochs, linear bs=512 lr=5
best accuracy: 94.47% for epoch 1000, trained for 1000 epochs, linear bs=512 lr=2.5
best accuracy: 94.46% for epoch 1000, trained for 1000 epochs, linear bs=512 lr=1.25
best accuracy: 94.39% for epoch 1000, trained for 1000 epochs, linear bs=512 lr=0.25
best accuracy: 94.50% for epoch 1000, trained for 1000 epochs, linear bs=256 lr=2.5
best accuracy: 94.40% for epoch 1000, trained for 1000 epochs, linear bs=128 lr=5
best accuracy: 94.41% for epoch 1000, trained for 1000 epochs, linear bs=128 lr=2.5
best accuracy: 94.45% for epoch 1000, trained for 1000 epochs, linear bs=128 lr=1.25
best accuracy: 94.48% for epoch 1000, trained for 1000 epochs, linear bs=128 lr=0.25
- [x] *6$: train the CIFAR10 dataset with resnet50, --batch_size=256, --lr=0.125, --temp=0.1.
best accuracy: 94.68% for last epoch, trained for 1000 epochs, linear bs=256
- [x] *7$: train the CIFAR10 dataset with resnet50, --batch_size=256, --lr=0.125, --temp=0.1.
best accuracy: 94.44% for epoch 3000, trained for 3000 epochs, linear bs=512 lr=5
- [x] *8$: train the CIFAR10 dataset with resnet50, --batch_size=128, --lr=0.005, --temp=0.07.
best accuracy: 94.51% for epoch 950, trained for 1000 epochs, linear bs=512 lr=5
best accuracy: 94.49% for epoch 900, trained for 1000 epochs, linear bs=512 lr=2.5
best accuracy: 94.46% for epoch 950, trained for 1000 epochs, linear bs=512 lr=1.25
best accuracy: 94.38% for epoch 900, trained for 1000 epochs, linear bs=512 lr=0.25
best accuracy: 94.49% for epoch 1000, trained for 1000 epochs, linear bs=128 lr=5
best accuracy: 94.49% for epoch 950, trained for 1000 epochs, linear bs=128 lr=2.5
best accuracy: 94.50% for epoch 950, trained for 1000 epochs, linear bs=128 lr=1.25
best accuracy: 94.45% for epoch 950, trained for 1000 epochs, linear bs=128 lr=0.25
- [x] *9s: train the DYB-Plankton with resnet50, --batch_size=32, --lr=0.0125, --temp=0.07 
DYB-Padded/train
Elapsed time: 4 days 2 hours
best accuracy: 93.42% head training bs=32, lr=0.0125
best accuracy: 93.59% head training bs=512, lr=0.25
- [x] *10s: train the DYB-Plankton with seresnext50, --batch_size=32, --lr=0.0125, --temp=0.07 
DYB-Padded/train
Elapsed time: 6 days 17 hours
best accuracy: 93.2%, head trained bs=512, lr=0.25 DYB-Padded/val
Best accuracy at epoch 86 is 89.95%, head trained bs=512, lr=0.25 DYB_val
- [x] *11s: train the DYB-Plankton with resnet50timm, --batch_size=32, --lr=0.0125, --temp=0.07 
DYB-Padded/train
Elapsed time: 4 days 12 hours
Best accuracy at epoch 84 is 93.48%, head training bs=512, lr=0.25 DYB-Padded/val
Best accuracy at epoch 76 is 90.29%, head training bs=512, lr=0.25 DYB_val
- [x] *12: train the DYB-Plankton with resnet50timm, --batch_size=32, --lr=0.0125, --temp=0.07
DYB-PlanktonNet/original
Elapsed time: 4 days 12 hours
Best accuracy at epoch 65: 94.61
- [] *13 train the DYB-Plankton with seresnext50timm, --batch_size=32, --lr=0.0125, --temp=0.07
DYB-PlanktonNet/original
Elapsed time: 6 days 18 hours
Best accuracy at epoch 67 is 94.49
- [x] *14: train the DYB-Plankton with resnet50timm, --batch_size=32, --lr=0.0125, --temp=0.07
DYB-PlanktonNet/original
Elapsed time: 4 days 12 hours
Best accuracy at epoch 65: 94.61



- [] *3$: train the CIFAR10 dataset with seresnext50, batch_size=128, --lr=0.2, --temp=0.07
- [] *4$: train the CIFAR10 dataset with resnet50, --batch_size=128, --lr=0.5, --temp=0.1.
best accuracy: 96.01% PAPER REFERENCE.
- [] *5$: train the CIFAR10 dataset with resnet50, --batch_size=512, --lr=0.5, --temp=0.1.
best accuracy: xx.xx% PERFORMING NOW
- [] *6$: train the CIFAR10 dataset with seresnext50 - maxpool, --batch_size=64, --lr=0.5, --temp=0.1.
best accuracy: 95.54%
- [] *7$: train the CIFAR10 dataset with seresnext50 - maxpool, --batch_size=64, --lr=0.25, --temp=0.1.
best accuracy: xx.xx% (IN PROCESS)
- [] *8$: train the CIFAR10 dataset with resnet50timm, --batch_size=128, --lr=0.5, --temp=0.1
best accuracy: 94.54% pretrained=True
best accuracy: 94.89% pretrained=False
- [] *9$: train the CIFAR10 dataset with resnet50pytorchCIFAR, --batch_size=128, --lr=0.5, --temp=0.1
best accuracy: 94.67% pretrained=False
- [] *10$: train the CIFAR10 dataset with resnet50 original + maxpool, --batch_size=128, --lr=0.5, --temp=0.1
best accuracy: 94.73% (RE-DO, MAXPOOL ORDER NOT IMPLEMENTED CORRECTLY, max pool before relu)
best accuracy: xx.xx% (max pool after relu, this is OK)
- [] *11$: train the CIFAR10 dataset with resnet50timm - maxpool, --batch_size=128, --lr=0.5, --temp=0.1
best accuracy: xx.xx%
- [] *12$: train CIFAR100 with resnet50, --batch_size 512, --lr=0.25, epochs max = 2000 (keeping ratios)
best accuracy: 72.62% vs 76.5%
- [] *$: train the DYBtrainCropped dataset with resnet50.
- [] *$: train the DYBtrainCropped dataset with seresnext50.
- [] *$: train the DYBtrainPadded dataset with resnet50.
- [] *$: train the DYBtrainPadded dataset with seresnext50.

### Elapsed time by type of training

* 1$: Elapsed time: 1 days 18 hours
* 3$: Elapsed time: 0 days 7 hours
```
python main_supcon.py --batch_size 128 --num_workers 8 --learning_rate 0.2 --lr_decay_epochs 10 --temp 0.07 --cosine
```
* 4$: Elapsed time: 1 days 18 hours
* 6$: Elapsed time: 2 days 18 hours
* 7$: 
* 8$: Elapsed time: 0 days 13 hours
```
python main_supcon.py --batch_size 128 --num_workers 8 --learning_rate 0.5 --lr_decay_epochs 10 --temp 0.1 --cosine
```
* 9$: Elapsed time: 0 days 13 hours
* 10$: Elapsed time: 0 days 12 hours
* 11$: 
* 12$: Elapsed time: 1 days 8 hours
* 13$: Elapsed time: 0 days 6 hours
```
python main_supcon.py --batch_size 128 --num_workers 8 --learning_rate 0.2 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0613, 0.0559, 0.0583" --std "0.1215, 0.1185, 0.1019" --dataset path --data_folder /home/dsosatr/tesis/DYBtrainCropped/
```
* 14$: Elapsed time: 1 days 8 hours
* 15$: Elapsed time: 0 days 6 hours
```
python main_supcon.py --batch_size 128 --num_workers 8 --learning_rate 0.2 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0361, 0.0326, 0.0357" --std "0.0906, 0.0882, 0.0774" --dataset path --data_folder /home/dsosatr/tesis/DYBtrainPadded/
```

ACCURACY CALCULATION:
```
python main_linear.py --batch_size 128 --num_workers 8 --learning_rate 5 --ckpt /home/dsosatr/tesis/cnnmodel/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_128_temp_0.1_trial_0_cosine_warm/last.pth
```

### Notes regarding training

[It is quite common for images with low resolution (e.g. 32x32 for CIFAR-10). In order to keep more spatial information before the average pool operator, the max pooling layer is removed. Notice that the filter-, stride- and padding size in the first convolution layer are also different from the conventional implementation. Again, this is to better preserve spatial information.](https://github.com/wvangansbeke/Unsupervised-Classification/issues/74)

[In general you should scale the learning rate linearly with the batch size, such that a smaller batch size corresponds with a smaller learning rate. This is only valid for sgd though, adam has different rules.](https://github.com/wvangansbeke/Unsupervised-Classification/issues/26)

### Reviewing logs after model training

In terminal, change path to where the log file exists, then:

	tensorboard --logdir=.

If getting an error when training the model about "tensorflow not installed" derived from tensorboard, just "pip uninstall tensorflow" and then reinstall tensorboard-logger.

## Git repository
Syncing local folder with remote repository:

	git fetch
	git merge origin/master

Syncing local changes with the remote server:

	git push

To remove the last commit in your local repository:

	git reset --hard HEAD~1

To force push this change to your remote repository:

	git push origin +HEAD

To see changes in a specific file:

	git diff <file>

## The seven rules of a great Git commit message

* Separate subject from body with a blank line
* Limit the subject line to 50 characters
* Capitalize the subject line
* Do not end the subject line with a period
* Use the imperative mood in the subject line
* Wrap the body at 72 characters
* Use the body to explain what and why vs. how

### Commits messages structured as follows:
[https://www.conventionalcommits.org/en/v1.0.0/](https://www.conventionalcommits.org/en/v1.0.0/)

[Commits style](https://chris.beams.io/posts/git-commit/)

### Variable and function names convention (PEP 8, official style guide for Python code)
* Variable names: my_variable, student_name, book_title
* Function names: calculate_average, get_student_grade, print_book_details
* Class names: MyClass, StudentRecord, BookDetails

## Other links
[SCL repository](https://github.com/HobbitLong/SupContrast)

[Building LeNet5 in Pytorch](https://youtu.be/0TtYx_XaXjA?si=rX2Pzy1NJQ5G0ZRU)
[, git repository](https://github.com/maciejbalawejder/Deep-Learning-Collection/tree/main/ConvNets/LeNet)
[, medium publication.](https://medium.com/nerd-for-tech/convolution-neural-network-in-pytorch-81023e7de5b9)

## Markdown preview in Visual Studio Code

To switch between views, press ⇧⌘V in the editor. You can view the preview side-by-side (⌘K V) with the file you are editing and see changes reflected in real-time as you edit.
