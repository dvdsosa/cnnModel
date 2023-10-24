# SEResNeXt trained with SCL

## TASKS

### DONE
- [x] Preprocessing: calculate mean and std for Z-Score formula.
- [x] Added Supervised Contrastive Learning (SCL) repository.
- [x] Manage error in the \_\_main\_\_ of main_supcon.py to send an email when it breaks.
- [x] Added DYB-PlanktonNet dataset to main_supcon.py.
	
### TODO next day
- [] Modify the resnet_big.py accordingly to the number of classes in DYB-PlanktonNet.
- [] CNN training.

## CNN Model

### Model training

	python main_supcon.py --batch_size 128 --num_workers 8 --learning_rate 0.2 --lr_decay_epochs 10 --lr_decay_rate 0.0001 --weight_decay 1e-4 --momentum 0.9 --temp 0.07 --cosine

### Reviewing logs after model training

In terminal, change path to where the log file exists, then:

	tensorboard --logdir .

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
