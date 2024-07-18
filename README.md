# ResNet50 trained with SCL

## TASKS

### DONE
- [x] Preprocessing: create training and testing sets, 01RandomSplitDataset.py file.
- [x] Preprocessing: calculate mean and std for Z-Score formula, 02MeanStdVariables.py file.
- [x] Added Supervised Contrastive Learning (SCL) repository.
- [x] Added notification by email when it ends or breaks.
- [x] Added DYB-PlanktonNet dataset to main_supcon.py.
- [x] Added SeResNext50 model to SCL repository.
- [x] Learn how to do the inference with this method.
- [x] Calculate performance metrics Acc (top-1), Recall, Precision, F1-Score.
- [x] Peform training with original size of 224 by 224.
- [x] Train the backbone with DYB-linearHead dataset (train, val, test folders).
- [x] Train the backbone with DYB-cosine dataset (train, test folders).
	
### TODO next day

- [] Compare the performance of the backbone trained using SeResNext50 versus ResNet50.
- [] Perform LUT prunning.


## CNN Model

### Training

#### Methodology

The backbone training is performend on a fixed number of epochs (1000). Later, we observe the loss decay via tensorboard, and pick the saved epochs where we start seeing a loss' plateau (in our case, from 700 to last). Next, we train a linear head over each one of these epochs while monitoring the validation performance. Thus, we select the weights were the validation accuracy is the highest, and with these weights, we evaluate the performance of the linear head using the testing set.

For DYB-linearHead dataset, **batch size 64**:

1. Backbone training (features extractor). Elapsed training time 3 days, 21 hours.
- The batch size was set to 64 to maximize the use of the GPU RAM while training, with the goal of minimizing the training time (let's see if this is true, performing now the training with batch size 32, let's see how log it takes...)

```bash
python main_supcon.py --batch_size 64 --num_workers 8 --learning_rate 0.016 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0418, 0.0353, 0.0409" --std "0.0956, 0.0911, 0.0769" --dataset path --data_folder /home/dsosatr/tesis/DYB-linearHead/train/ --size 224
```

2. Head training (classifier). It is necessary to iterate over every backbone weights (in our case we iterate from epoch 700 to last):

```bash
./main_linear_loop.sh
```

---

For DYB-linearHead dataset, **batch size 32**:

1. Backbone training (features extractor). Elapsed training time x days, x hours.
- The learning rate of 0.016 was selected as a linear relation between the original batch size of 1024 and learning rate of 0.5; thus dividing 0.5/32 = 0.0156 rounded to 0.016. 

```bash
python main_supcon.py --batch_size 32 --num_workers 8 --learning_rate 0.016 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0418, 0.0353, 0.0409" --std "0.0956, 0.0911, 0.0769" --dataset path --data_folder /home/dsosatr/tesis/DYB-linearHead/train/ --size 224
```

--- 

For DYB-cosine dataset, **batch size 32**:

1. Backbone training (features extractor):
```bash
python main_supcon.py --batch_size 32 --num_workers 8 --learning_rate 0.016 --lr_decay_epochs 10 --temp 0.07 --cosine --mean "0.0419, 0.0355, 0.0410" --std "0.0959, 0.0913, 0.0771" --dataset path --data_folder /home/dsosatr/tesis/DYB-original/train/ --size 224
```

#### Results

Resnet50timm, batch_size = 64, learning_rate = 0.016, dataset DYB-linearHead, head trained using the frozen backbone on epoch 800 and validated with the DYB-original/val folder. 17 jul 2024 20:03.

| ckpt number | accuracy | epoch |
| ----------- | -------- | ----- |
| last.pth    | 95.10%   | 61    |
| 1000.pth    | 95.02%   | 61    |
| 950.pth     | 95.03%   | 51    |
| 900.pth     | 94.89%   | 53    |
| 850.pth     | 94.88%   | 66    |
| 800.pth     | 95.17%   | 49    |
| 750.pth     | 94.64%   | 70    |
| 700.pth     | 94.64%   | 21    |

Finally, we test the accuracy of the backbone epoch 800 weights, and the head epoch 800 weights, with the DYB-original/test dataset.

| Accuracy | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| 94.86%   | 93.65%    | 90.26% | 91.43%   |

### Notes regarding training

[It is quite common for images with low resolution (e.g. 32x32 for CIFAR-10). In order to keep more spatial information before the average pool operator, the max pooling layer is removed. Notice that the filter-, stride- and padding size in the first convolution layer are also different from the conventional implementation. Again, this is to better preserve spatial information.](https://github.com/wvangansbeke/Unsupervised-Classification/issues/74)

[In general you should scale the learning rate linearly with the batch size, such that a smaller batch size corresponds with a smaller learning rate. This is only valid for sgd though, adam has different rules.](https://github.com/wvangansbeke/Unsupervised-Classification/issues/26)

## Remarks regarding environment configuration:

### Create a new python environment:

In Ubuntu 24.04 this is necessary to install python packages using pip:
```bash
python3 -m venv myenvTesis
source myenvTesis/bin/activate
pip install package_name

source deactivate
```

### numpy version < 2.0.0

ATTENTION! numpy version < 2.0.0 needed, if not, it will fail with the following error:
> OverflowError: Python integer -20 out of bounds for uint8

```bash
pip install numpy==1.26.4
```

### Analyzing loss curve after model training

In terminal, change path to where the log file exists, then:

```bash
pip uninstall numpy
pip install numpy==1.26.4
source envTensorboard/bin/activate
tensorboard --logdir=.
source deactivate
```

If getting an error when training the model about "tensorflow not installed" derived from tensorboard, just "pip uninstall tensorflow" and then reinstall tensorboard-logger.

## Git repository
Add local repo to Bitbucket and Github accounts:

	git remote add origin git@bitbucket.org:davidsosatrejo/cnnmodel.git
	git remote set-url --add origin git@github.com:dvdsosa/cnnModel_private.git
	git remote -v
	git push origin master

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

To remove a single file in all commits:

	pip install git-filter-repo
	git filter-repo --path path/to/file --invert-paths
	git push origin --force --all 

## The seven rules of a great Git commit message

* Separate subject from body with a blank line
* Limit the subject line to 50 characters
* Capitalize the subject line
* Do not end the subject line with a period
* Use the imperative mood in the subject line
* Wrap the body at 72 characters
* Use the body to explain what and why vs. how

### Commits messages structured as follows:
- [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
- [Commits style](https://chris.beams.io/posts/git-commit/)

### Variable and function names convention (PEP 8, official style guide for Python code)
* Variable names: my_variable, student_name, book_title
* Function names: calculate_average, get_student_grade, print_book_details
* Class names: MyClass, StudentRecord, BookDetails

## Other links
- [SCL repository](https://github.com/HobbitLong/SupContrast)
- [Building LeNet5 in Pytorch](https://youtu.be/0TtYx_XaXjA?si=rX2Pzy1NJQ5G0ZRU)
- [LeNet5 git repo](https://github.com/maciejbalawejder/Deep-Learning-Collection/tree/main/ConvNets/LeNet)
- [CNN using Pytorch](https://medium.com/nerd-for-tech/convolution-neural-network-in-pytorch-81023e7de5b9)

## Markdown preview in Visual Studio Code

To switch between views, press ⇧⌘V in the editor. You can view the preview side-by-side (⌘K V) with the file you are editing and see changes reflected in real-time as you edit.
