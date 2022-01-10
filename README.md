# Image Classification and Visualizations with Convolutional Neural Networks --- ImageNet10

One challenge of building a deep learning model is to choose an architecture that can learn the features in the dataset without being unnecessarily complex. This project involves building a CNN and training it on ImageNet10. I will use a method of architecture development called “single-batch training”, in which we cumulatively build a network architecture which can overfit a single training batch. A model which overfits performs very well on training data but generalises poorly on data it has not been trained on. If the model can easily overfit a single training batch, we know its architecture is complex enough to be able to learn the features present in the training data. Then we move on to training on the complete training set and adjust for the overfitting via regularisation.

This project will use a subset of images from the ImageNet dataset, which contains 9000 images belonging to ten object classes. We will refer to the dataset as ImageNet10. The images can be found in a Git repository: https://github.com/MohammedAlghamdi/imagenet10.git

In the part of image classification, I firstly use only one batch of the training data, and part or all of the validation data. I also adjust Adjust the network by adding a combination of convolutional and fully-connected layers, ReLU, max-pool, until the training and validation loss show that the model is overfitting the training batch.

Secondly, I train the model on the complete training dataset, and use the complete validation set to determine when to stop training. I experiment with some form of regularization such as dropout or data augmentation to overfit the training data. Besides, two confusion matrices, one for the training set and one for the validation set is displayed to show the model performance.

![graph](https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/Image%20Classification/training%20and%20validation%20loss.png)

At last, I use the best finetuned model on the unlabeled test data and generate predictions.

In the part of CNN visualization, it involves to extract the filters from a given layer of the AlexNet model. In this way, extracting and visualizing feature maps is available which are the result of the filter kernels applied to the convolutional layer input.

