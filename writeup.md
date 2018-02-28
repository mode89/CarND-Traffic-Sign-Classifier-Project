# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[classes]: ./images/classes.png "Classes"
[class_distribution]: ./images/class_distribution.png "Class Distribution"
[preprocessing]: ./images/preprocessing.png "Preprocessing"
[downloaded]: ./images/downloaded.png "Downloaded"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is the set of random images from the training dataset. Each image
represent one of the classes:

![Classes][classes]

Here is the histogram showing how many images belongs to each of the classes
in the training, validation and testing datasets:

![Class distribution][class_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale, because it
makes the model smaller. As we can see from the previous section, non of the
presented classes requires color information to be distinguished from other
classes. Then I normalize images, so that input data has mean zero and equal
variance.

Here is an example of a traffic sign image after preprocessing (from left to
right: raw image, grayscale, normalized):

![Preprocessing][preprocessing]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I've taken the vanilla LeNet-5 model and changed the number of outputs of
the final layer, since in our experiment we have different amount of
classes. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Normalized grayscale image            | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6    |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6                   |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16                    |
| Flatten               |                                               |
| Fully connected		| inputs 400, outputs 120                       |
| RELU                  |                                               |
| Fully connected		| inputs 120, outputs 84                        |
| RELU                  |                                               |
| Fully connected       | inputs 84, outputs 43                         |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, because it's more
sophisticated than SGD and considered as a good choice for starter. The
batch size, I selected, is 128. I used exponentially decaying learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.947
* test set accuracy of 0.924

As a starter point, I chosen LeNet-5 archicture, because it was suggested by
the course lectures. I've started with RGB images. The initial network
architecture needed to be modified, so that I could be feed it RGB images.
Also I had to modify the size of the output layer, since in our task we have
the different amount of classes. This model wasn't able to reach 0.93
accuracy, that's why later I've implemented grayscale preprocessing and got
back to the input layer of size 32x32x1. I tried the constant learning
rate of 0.0001, 0.001 and 0.01, but it didn't help to reach the accuracy of
0.93. Then I switched to the learning rate exponanentially decaying from
0.01 to 0.001, and the network was able to reach an accuracy above 0.93 in
10 training epochs. I think that a convolution layer work well with this
problem, because it works as a feature detector, and in conjunction with
max-pooling can detect features of different levels, thus building up a set
of 2D multi-level features describing an image. The high training accuracy
and the low validation/testing accuracy indicate that the model is
overfitting. I think that a dropout layer might help with eliminating
overfitting. It forces a neural network to learn more robust features that
are useful in conjunction with many different random subsets of the other
neurons.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Downloaded Images][downloaded]

The first three images might be difficutl to classify because they have
watermarks obscuring signs. I modified the original images by cutting out
part of image not related to sign, and leaving square region containing
image of sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image                 | Prediction                                    |
|:---------------------:|:---------------------------------------------:| 
| Piority Road          | Priority Road                                 |
| Stop                  | Stop                                          |
| Keep Right            | Keep Right                                    |
| Children Crossing     | Children Crossing                             |
| Yield                 | Yield                                         |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 16th
cell of the Ipython notebook.

For the first image, the model is sure that this is a priority road sign
(probability of 1.0), and the image does contain a priority road sign. The
top five soft max probabilities were:

| Probability   | Prediction                                |
|:-------------:|:-----------------------------------------:|
| 1.0           | Priority Road                             |
| 5e-09         | No Vehicles                               |
| 2e-09         | Keep Right                                |
| 4e-10         | Children Crossing                         |
| 3.5e-10       | End of all limits                         |

For the second image, the model is sure that this is a stop sign
(probability of 0.97), and the image does contain a stop sign. The top five
soft max probabilities were:

| Probability   | Prediction                                |
|:-------------:|:-----------------------------------------:|
| 0.97          | Stop                                      |
| 0.03          | Priority Road                             |
| 9e-04         | Right-of-way at the next intersection     |
| 1e-04         | Keep Right                                |
| 4e-05         | Beware of ice/snow                        |

For the third image, the model is sure that this is a keep right sign
(probability of 1.0), and the image does contain a keep right sign. The top
five soft max probabilities were:

| Probability   | Prediction                                |
|:-------------:|:-----------------------------------------:|
| 1.0           | Keep Right                                |
| 6e-14         | Priority Road                             |
| 1e-14         | Speed limit 60 km/h                       |
| 2e-18         | Turn left ahead                           |
| 7e-19         | Children crossing                         |

For the fourth image, the model is sure that this is a children crossing
sign (probability of 1.0), and the image does contain a children crossing
sign. The top five soft max probabilities were:

| Probability   | Prediction                                |
|:-------------:|:-----------------------------------------:|
| 1.0           | Children crossing                         |
| 3e-4          | Right-of-way at the next intersection     |
| 2e-4          | Dangerous curve to the right              |
| 3e-5          | End of no passing                         |
| 9e-6          | Slippery road                             |

For the fifth image, the model is sure that this is a yield sign
(probability of 1.0), and the image does contain a yield sign. The top five
soft max probabilities were:

| Probability   | Prediction                                |
|:-------------:|:-----------------------------------------:|
| 1.0           | Yield                                     |
| 2e-22         | Turn right ahead                          |
| 7e-26         | Turn left ahead                           |
| 1.5e-26       | Speed limit 60 km/h                       |
| 1.3e-26       | No passing                                |
