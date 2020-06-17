# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Visualization.png "Visualization"
[image2]: ./examples/traindata.png "traindata"
[image3]: ./examples/Augmented_data.png "Augmented_data"
[image4]: ./examples/web_images.png "web_images"
[image5]: ./examples/performance.png "performance"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/7ee8d0d4-561e-4101-8615-66e0ab8ea8c8/concepts/a96fb396-2997-46e5-bed8-543a16e4f72e#)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12360
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The images show bar graph of training data and random training images samples.

![Visualization][image1]
![traindata][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Larger data set is important for a good Model to predict the class correctly. The provided training data set has fewer number of samples for a few classes. Therefore firstly I augmented the training data set by adding slightly modified images of the existing training images. Class IDs are sorted based on number of training samples corresponding to each class and then first 15 classes with least number of samples are chosen. The chosen images are then rotated slightly to generate modified image for training.

Further the images in augmented data set are then converted to grayscale. It reduces computation load without compromising on features. The grayscaled images are then normalized about zero. Normalised data set is uniformly in all the dimensions and it helps in optimising the parameters faster and effectively.

Last step of preprocessing is to shuffle the normalized data to train the model effectively for maximum number of classes in a single batch.
The new augmented data set is depicted in bar graph shown below
![Augmented_data][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The LeNet-5 model is used as the base architecture. The number of parameters in hidden layers are modified and tuned in iterative process. It was noted that the traffic signs have different shapes and marking and thus have several different features like edges, curves,contrast etc. Therefore number of parameters are increased in iterative fashion to achieve 97% validation accuracy.

First layer is responsible for edges. Therefore number of filters is chosen and tried to cater edges and curves of markings. Hidden Layer 1 : 32x32x1 -> 5x5 convolution with 20 filters -> 28x28x20 -> ReLU -> maxPooling 2x2 -> 14x14x20

Second layer is responsible for basic underlying shapes of signs. Therefore number of filters is chosen and tried to cater general shapes Hidden Layer 2 : 14x14x20 -> 5x5 convolution with 80 filters -> 10x10x80 -> ReLU -> maxPooling 2x2 -> 5x5x80

Size of Connected layers were chosen and tried to train the model differentiate between closely resembling signs. Fully connected layer 1 : flatten(5x5x80) -> 2000 ->Dropout layer while dropping probability 50%->240

Fully connected layer 2 : 240 -> Dropout layer while dropping probability 50%->150

Fully connected layer 3 : 150 -> 43

The number of parameters were initialised considering all the factors explained above. However final number of parameters is fixed after several iterations.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x80	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x80   				|
| Flatten	        	| output 2000                   				|
| Fully connected		| 2000 input, output 240  						|
| Fully connected		| 240 input, output 150							|
| Softmax               | 150, output 43								|

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Adamoptmizer is used with training batch size 64.

Target is to achieve validation accuracy of 97% or above. The pipeline stops training the model if the target accuracy is achieved or epoch value hits 100, whichever happens first.

The learning rate is kept .00035. It slows down the learning process a bit but helps in achieving the target.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 
* validation set accuracy of 97%
* test set accuracy of 95%

* As mentioned earlier I chose the well known and reliable Lenet-5 architecture as base. 
* Basic architecture did not have sufficient parameters to capture the features of traffic signs completly. I did not get desired  validation accuracy of 97% with basic LeNet5 architecture.
* I modified the architecture by inceasing the number of parametrs in every layer while keeping optimum number of parameters.
The learning rate was kept lower on expanse of computaion load to achieve the resolution to achiev the desired validation accuracy of 97%.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 7 German traffic signs that I found on the web:

![web_images][image4] 

Chosen images address to the challenge that two signs could be mistaken.For example signs of speed limit 50 and speed limit 60 may have almost similar feature values, which may cause an under trained model to make mistakes. Similarly an under trained model can make mistake in predicting signs between man at work and bumpy road signs. Even with validatin accuracy of 95% , model can make mistake. Therefor I tageted 97% of validation accuracy to make the model successful and I tuned model parameters accordingly. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%.
The picture below shows the input images and corresponding softmax probablity of top five candidates.
![performance][image5]


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For all the seven images the model predicted with almost 100% accuracy .

The accuracy of test set data is 95%. Whereas Accuracy of web images test set is 100%.

The top 5 probalities in first column is almost 1 and correctly predicts the corresponding class ID.

![performance][image5]

TopKV2(values=array(  [[  9.99997854e-01,   1.97560234e-06,   1.03495218e-07, 3.52766705e-08,   3.23495932e-08],
                       [  1.00000000e+00,   4.43811849e-15,   8.05913892e-16, 1.68926150e-16,   4.44365106e-18],
                       [  9.98067915e-01,   1.68942427e-03,   1.75444555e-04, 6.62640959e-05,   5.79240918e-07],
                       [  1.00000000e+00,   6.63996746e-10,   4.14149888e-13, 6.62669111e-15,   3.00422734e-17],
                       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00],
                       [  1.00000000e+00,   3.96173067e-10,   5.76797074e-13, 1.40772525e-17,   7.65688171e-18],
                       [  1.00000000e+00,   6.89140970e-21,   6.27488552e-23, 2.52903186e-24,   3.62659388e-26]], dtype=float32), 
       indices=array( [[ 3, 34, 11, 35, 12], 
                       [34, 40, 35, 28, 30],
                       [25, 20, 39, 22, 17],
                       [ 1,  5,  8,  4,  2],
                       [38,  0,  1,  2,  3],
                       [18, 27, 11, 26, 40],
                       [11, 18, 27, 30, 23]], dtype=int32))


