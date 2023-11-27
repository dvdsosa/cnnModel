# SEResNeXt trained with SCL

## TASKS

### DONE
- [x] Preprocessing: create training and testing sets, 01RandomSplitDataset.py file.
- [x] Preprocessing: calculate mean and std for Z-Score formula, 02MeanStdVariables.py file.
- [x] Added Supervised Contrastive Learning (SCL) repository.
- [x] Manage error in the \_\_main\_\_ of main_supcon.py to send an email when it breaks.
- [x] Added DYB-PlanktonNet dataset to main_supcon.py.
	
### TODO next day
- [] Learn how to do the inference with this method.
- [] Employ SEResNeXt as the model instead default ResNest50.
- [] Peform training with original size of 224 by 224.

## CNN Model

### Model training

	python main_supcon.py --batch_size 16 --num_workers 8 --learning_rate 0.2 --lr_decay_epochs 10 --lr_decay_rate 0.0001 --momentum 0.9 --temp 0.07 --cosine --mean "0.0613, 0.0559, 0.0583" --std "0.1215, 0.1185, 0.1019" --dataset path --data_folder /home/dsosatr/tesis/DYBtrainCropped/ --size 224

### Elapsed time by type of training

TEST DYBtrainCropped

	--batch_size 128, --size 32, --data_folder DYBtrainCropped, 1days 16hours. erased.
	--batch_size 64, --size 56, --data_folder DYBtrainCropped, 4days 2hours. finished
	above stored on --> SupCon_path_resnet50_lr_0.2_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm
	--batch_size 16, --size 224, --data_folder DYBtrainCropped

TEST CIFAR10

	python main_supcon.py --batch_size 64 --num_workers 8 --learning_rate 0.2 --lr_decay_epochs 10 --lr_decay_rate 0.0001 --momentum 0.9 --temp 0.07 --cosine
	Script execution completed! Elapsed time: 1 days 19 hours

### Reviewing logs after model training

In terminal, change path to where the log file exists, then:

	tensorboard --logdir=.

If getting an error when training the model about "tensorflow not installed" derived from tensorboard, just "pip-autoremove tensorboard_logger -y" and then "pip install tensorboard-logger"

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

## Markdown preview in Visual Studio Code

To switch between views, press ⇧⌘V in the editor. You can view the preview side-by-side (⌘K V) with the file you are editing and see changes reflected in real-time as you edit.
