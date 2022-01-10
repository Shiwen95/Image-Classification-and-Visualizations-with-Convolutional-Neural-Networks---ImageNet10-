# Image Classification and Visualizations with Convolutional Neural Networks --- ImageNet10

One challenge of building a deep learning model is to choose an architecture that can learn the features in the dataset without being unnecessarily complex. This project involves building a CNN and training it on ImageNet10. I will use a method of architecture development called “single-batch training”, in which we cumulatively build a network architecture which can overfit a single training batch. A model which overfits performs very well on training data but generalises poorly on data it has not been trained on. If the model can easily overfit a single training batch, we know its architecture is complex enough to be able to learn the features present in the training data. Then we move on to training on the complete training set and adjust for the overfitting via regularisation.

This project will use a subset of images from the ImageNet dataset, which contains 9000 images belonging to ten object classes. We will refer to the dataset as ImageNet10. The images can be found in a Git repository: https://github.com/MohammedAlghamdi/imagenet10.git

In the part of image classification, I firstly use only one batch of the training data, and part or all of the validation data. I also adjust Adjust the network by adding a combination of convolutional and fully-connected layers, ReLU, max-pool, until the training and validation loss show that the model is overfitting the training batch.

Secondly, I train the model on the complete training dataset, and use the complete validation set to determine when to stop training. I experiment with some form of regularization such as dropout or data augmentation to overfit the training data. Besides, two confusion matrices, one for the training set and one for the validation set is displayed to show the model performance.

![graph](https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/Image%20Classification/training%20and%20validation%20loss.png)

![graph](https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/Image%20Classification/confusion%20matrix.png)

At last, I use the best finetuned model on the unlabeled test data and generate predictions.

In the part of AlexNet visualization, it involves to extract the filters from a given layer of the AlexNet model. In this way, extracting and visualizing feature maps is available which are the result of the filter kernels applied to the convolutional layer input.

|      | Filter | Feature map | Brief explanation |
|------------|-------------|-------------|-------------|
| Early layer | <img src="https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/CNN%20Visualizations/Early%20Filter.png" width="300"> | <img src="https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/CNN%20Visualizations/Early%20Feature.png" width="250"> | I choose the first layer and its 41st filter to display. This filter is aim to extract specific edge features having low weights in the center and high weights around the center. In this way, differences are accentuated and constant area left untouched. It can be easily proved by the feature map that two cats’ outlines are obvious. And some other minor features including shapes of eyes, beard, and patterns of those cats are clear as well. To sum up, a pixel’s neighborhood contains information about its intensity.|
| Intermediate layer | <img src="https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/CNN%20Visualizations/Intermediate%20Filter.png" width="300"> | <img src="https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/CNN%20Visualizations/Intermediate%20Feature.png" width="250"> | I choose the second layer and its 57th filter to display. This filter is aim for linear edge detection having relatively low weights in the upper half while high weights in the second half. To be specific, it changes the pixel based on its neighbor beneath. It can be easily proved in the feature map that two cats’ strong linear features are obvious. The filter identifies not only ears’ curly shape but also the linear edge of carton. |
| Deep layer | <img src="https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/CNN%20Visualizations/Deep%20Filter.png" width="300"> | <img src="https://github.com/Shiwen95/Image-Classification-and-Visualizations-with-Convolutional-Neural-Networks---ImageNet10-/blob/main/CNN%20Visualizations/Deep%20Feature.png" width="250"> | I choose the fourth layer and its 57th filter to display. This filter is aim to change pixels depending on the bottom right corner where there is a high weight. It is highly focus on small scale of feature. It can be easily proved in the feature map that two cats’ scales are obvious. Besides, the edge of carton which is detected in the second layer is discarded in this layer. |

It is found that AlexNet is designed to output feature maps by using filters in different layers step by step. I would describe the differences of filters and the feature maps between different layers below. In the first layer, filters are 11-by-11 matrices. They can reduce noises such as shadows and constant areas on the backgrounds. They can also sharpen edges and detect differences according to image brightness changes. The most important thing for the feature maps is images are mainly identified for its general features including more details of textures, patterns and so on. Besides, feature maps change from original 224*224 resolution to 55*55 resolution. In the intermediate layer, the size and the task of filters have changed. By reducing size to 5*5 or 3*3, filters could reduce low-level noises and retain more informative pieces. How the filters work changes the output known as feature maps. Accordingly, we can notice that most of the feature maps includes thick edge elements and corners with few noises. And their resolution continues going down to 27*27 or 13*13. In the layer towards the end of the network, all of the filters and feature maps end up with the shape of 3*3 and 13*13 respectively. The feature maps remain a set of curved line segments termed edges. It is because they are filtered for the most important features like the outline of cat and football. In conclusion, the deeper the network, the smaller the image resolution. Additionally, features being extracted become more and more typical with the depth of model.
