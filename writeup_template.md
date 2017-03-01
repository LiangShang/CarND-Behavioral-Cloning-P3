#**Behavioral Cloning** 


**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup/model.png "Model Visualization"
[image2]: ./writeup/test.jpg "Recovery Image"
[image3]: ./examples/placeholder_small.png "Recovery Image"


###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I use the Nvidia pipeline as my model, as well as some changes on it.

It firstly has a lambda layer for normalization, followed by a cropping layer to trim the useless parts of the image, such as the sky and trees.

It also consists of 5 convolution neural networks. The first three are with 5x5 filter sizes and 2x2 strides, the depths of which are 24, 36 and 48 respectively. The following 2 networks are with 3x3 filters and 1x1 strides(default value). The depths of them are both 64. All the 5 networks have relu as the activation function, introducing the nonlinearity.

After the convolution layer are the fully connected layers, which are flatten and dense of 100, 50, 10 and 1. All of them are followed by a dropout, with a posibility of 0.3, which ensures the model is unlikely to overfitting.

The overall view of the model is as below.

![alt text][image1]



####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting, which is mentioned in the above section. 

The model was trained and validated on different data sets to ensure that the model was not overfitting, with parameter `validation_split=0.2` in function `model.fit()`  The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

I used the data provided by the course to train the model. I used the flipped image to reduce the left-turning bias. I only use the center image for training and cross validation. Below is an example of the image 

![alt text][image2]


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to come up with the model, train it, run with simulator, find the weaken part (alomost the turning parts), refine the model and loop. 

My first step was to use the lenet5 network. I thought it would be a good choice since it performs good in some image related classification and it might also work for the regression problem. But I was wrong, the predict of that model is not as good as I thought,

So the alternative model I used is a convolution neural network model similar to the Nvidia pipeline. I thought this model might be appropriate because it is the advise from the helpful guide.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I firstly find the my training loss is less than the cross validation loss so I think there would be an overfitting.

To combat the overfitting, I add the dropouts so that the training loss is similar to the cross validation loss, which means the overfitting is reduced, and as a matter of fact, the car runs better in the simulator.

The final epoch has the loss of 

`loss: 0.0100 - val_loss: 0.0096`


When I train the model, I shuffle the data randomly, putting 20% of the data into a validation set. Also from the helpful guide, about 5 epoch is enough for the training, so I set training epoch to 5 and the loss of training and cross validation rarely changes in the 4th and 5th epoch, which saves time.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


