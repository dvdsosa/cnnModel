# SEResNeXt trained with SCL

## DONE
- [x] Preprocessing: calculate the mean and standard deviation values used for Z-Score Normalization in main_supcon.py. This is performend on 01ZScoreMeanStd.py file.
- [ ] Added Supervised Contrastive Learning (SCL) repository.
	
## TODO next day
- CNN training.

## Model as a feature extractor

If you want to use a **Convolutional Neural Network (CNN)** as a feature extractor, you can proceed as follows:

1. **Training the CNN**: Initially, you need to train the CNN with all the layers, including the Fully Connected (FC) layer, using a labeled dataset relevant to your task. This training process helps the network learn meaningful representations of the input images and optimize the weights and biases of all layers.

2. **Feature Extraction**: Once the CNN is trained, you can remove the FC layer and use the remaining layers as a feature extractor. The convolutional, activation, pooling, batch normalization, and dropout layers can be retained for this purpose . By feeding an image through the network, you can obtain a vector of extracted features from one of the intermediate layers.

3. **Using Extracted Features**: The extracted features can then be used as input to another model or algorithm for further processing or analysis. For example, you can use these features as input to a support vector machine (SVM), a random forest classifier, or any other machine learning model that suits your specific task.

By training the CNN with all layers and then removing the FC layer for feature extraction, you leverage the power of pre-trained convolutional layers while customizing the final layers for your specific needs. This approach can save computational resources and time compared to training an entire CNN from scratch.

Please note that when using a pre-trained CNN as a feature extractor, it's important to freeze the weights of the convolutional layers to prevent them from being updated during subsequent training. Only the additional layers added after feature extraction should be trained on your specific dataset.

I hope this clarifies how to proceed when using a CNN as a feature extractor! Let me know if you have any further questions.