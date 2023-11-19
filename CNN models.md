# SEResNeXt trained with SCL

## TASKS

### DONE
- [x] Preprocessing: create training and testing sets, 01RandomSplitDataset.py file.
- [x] Preprocessing: calculate mean and std for Z-Score formula, 02MeanStdVariables.py file.
- [x] Added Supervised Contrastive Learning (SCL) repository.
- [x] Manage error in the \_\_main\_\_ of main_supcon.py to send an email when it breaks.
- [x] Added DYB-PlanktonNet dataset to main_supcon.py.
	
### TODO next day
- [x] Save the training and test set features into a SQL database.
- [-] Calculate the threshold for the posterior LUT pruning. EN TEST, UNA VEZ QUE SE VEA QUE SE CALCULA, VOLCAR EL TEST-SET COMPLETO A UNA DATABASE Y RE-CALCULAR EL THRESHOLD, PUEDE LLEVAR 1 SEMANA DE CÁLCULO EN CPU.
- [] Prune the dataset features according to Yang et al. paper supplementary material.
- [] Learn how to do the inference with this method.
- [] Employ SEResNeXt as the model instead default ResNest50.
- [] Peform training with original size of 224 by 224.

## Model as a feature extractor

If you want to use a **Convolutional Neural Network (CNN)** as a feature extractor, you can proceed as follows:

1. **Training the CNN**: Initially, you need to train the CNN with all the layers, including the Fully Connected (FC) layer, using a labeled dataset relevant to your task. This training process helps the network learn meaningful representations of the input images and optimize the weights and biases of all layers.

2. **Feature Extraction**: Once the CNN is trained, you can remove the FC layer and use the remaining layers as a feature extractor. The convolutional, activation, pooling, batch normalization, and dropout layers can be retained for this purpose . By feeding an image through the network, you can obtain a vector of extracted features from one of the intermediate layers.

3. **Using Extracted Features**: The extracted features can then be used as input to another model or algorithm for further processing or analysis. For example, you can use these features as input to a support vector machine (SVM), a random forest classifier, or any other machine learning model that suits your specific task.

By training the CNN with all layers and then removing the FC layer for feature extraction, you leverage the power of pre-trained convolutional layers while customizing the final layers for your specific needs. This approach can save computational resources and time compared to training an entire CNN from scratch.

Please note that when using a pre-trained CNN as a feature extractor, it's important to freeze the weights of the convolutional layers to prevent them from being updated during subsequent training. Only the additional layers added after feature extraction should be trained on your specific dataset.

I hope this clarifies how to proceed when using a CNN as a feature extractor! Let me know if you have any further questions.

## Backbone vs Projection head

In the context of a Convolutional Neural Network (CNN), the terms "backbone" and "projection head" refer to different parts of the network that have specific roles.

1. **Backbone**: The backbone is the main part of the CNN that is responsible for extracting features from the input data. It is typically composed of several layers of convolutions, pooling, and non-linear activations. The backbone transforms the raw input data (like an image) into a high-dimensional feature representation. This feature representation captures the essential characteristics of the input data that are relevant for the task at hand (like image classification or object detection).

2. **Projection Head**: The projection head is a component that is added to the end of the backbone. Its role is to map the high-dimensional feature representation produced by the backbone to a lower-dimensional space. The projection head is typically a small neural network itself, often composed of one or more fully connected layers. The purpose of the projection head is to make the feature representation more suitable for the downstream task. For example, in contrastive learning, the projection head maps the features to a space where similar images are close together and dissimilar images are far apart.

Here's a simplified diagram of a CNN with a backbone and projection head:

```
Input Data
    |
    V
[Backbone CNN]
    |
    V
[High-Dimensional Features]
    |
    V
[Projection Head]
    |
    V
[Low-Dimensional Features]
```

In this diagram, the input data is passed through the backbone CNN to produce high-dimensional features. These features are then passed through the projection head to produce low-dimensional features that are used for the downstream task. I'll try to create a graphical representation of this.

## Pytorch help

[Pytorch Transforms](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html)

[TRANSFORMING AND AUGMENTING IMAGES](https://pytorch.org/vision/main/transforms.html)

[Astronaut gallery example from above documentation](https://github.com/pytorch/vision/tree/main/gallery/)

[Read more about fix size mismatch](https://discuss.pytorch.org/t/how-to-fix-size-mismatch-pretrained-model-for-large-input-image-sizes/144025/2)