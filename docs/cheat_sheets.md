# Cheat Sheets

## GIT

### Repository management
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

### The seven rules of a great Git commit message
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

## Variable and function names convention (PEP 8, official style guide for Python code)
* Variable names: my_variable, student_name, book_title
* Function names: calculate_average, get_student_grade, print_book_details
* Class names: MyClass, StudentRecord, BookDetails

## Markdown preview in Visual Studio Code
To toggle between the markdown editor and its preview, press `⇧⌘V`. For a side-by-side view, use `⌘K V`. This allows you to see real-time updates in the preview as you edit the markdown file.
