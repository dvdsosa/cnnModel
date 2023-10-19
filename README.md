# SEResNeXt trained with SCL

## DONE
- [x] Preprocessing: calculate mean and std for Z-Score formula.
- [ ] Added Supervised Contrastive Learning (SCL) repository.
	
## TODO next day
- CNN training.

## Model training

	python main_supcon.py --num_workers 8 --batch_size 128 --learning_rate 0.5 --temp 0.07 --cosine

## Git repository
Syncing local folder with remote repository:

	git fetch
	git merge origin/master

Syncing local changes with the remote server:

	git push

To undo git add before a commit (unstage all changes):

	git reset


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

## Other links
[SCL repository](https://github.com/HobbitLong/SupContrast)

## Markdown preview in Visual Studio Code

To switch between views, press ⇧⌘V in the editor. You can view the preview side-by-side (⌘K V) with the file you are editing and see changes reflected in real-time as you edit.
