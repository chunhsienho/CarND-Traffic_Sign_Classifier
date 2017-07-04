#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code]
https://github.com/chunhsienho/CarND-Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the cells 1,2 of the IPython notebook 

The size of training set is 
* Number of training examples = 27839
The size of the validation set is ?
* Number of valid examples = 6960
The size of test set is ?
* Number of testing examples = 12630
The shape of a traffic sign image is ?
* The shape of a traffic sign image is 32x32x1 (as i use grayscale)
The number of unique classes/labels in the data set is ?
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

The code for this step is contained in the third code cell of the IPython notebook.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

There are two idea that use in the dataset preprocessing.

1.Normalize all vectors to the same scale (-1.0~1.0) since its RGB pixels, I simply used X_train_normalized = (X_train - 128)/128. The reason why i use this to reduce the distribution of the data. With a wider distribution, it would be harder to use a singlar learning rate.

2. I convert all the picture into grayscale so that i did not have to process the different layer for RGB. This method used in "Traffic Sign Recognition with Multi-Scale Convolutional Networks". By using the method mentioned in this paper, I could modified the LaNet5 in Lab to be my tensowerflow struture. 









####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description								| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x1 image   								| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| 2x2 Max pooling		| 2x2 stride,outputs 14x14x6					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| 2x2 Max pooling		| 2x2 stride, outputs 5x5x16					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Flatten layers		| 1x1x400 -> 400 and 5x5x16 -> 400 				|
| Concatenate  layers	| 400+400->800  								|
| Dropout				|												|
|Fully connected layer 	|800 in 43 out  								|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

batch size: 128
epochs: 120
learning rate: 0.001
mu: 0
sigma: 0.1
dropout keep probability: 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


If a well known architecture was chosen:
* What architecture was chosen?
The citation of the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks". I believe it could be a good architecture.
* Why did you believe it would be relevant to the traffic sign application?
Since the name is discussing about the german traffic sign, I think it would be very useful.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
training:
validation :0.99
test set :0.93

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


Stop sign, Speed limit(30km/h),Priority road,Keep right,Speed limit (60km/h),General Caution

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Priority road			| Priority road									|
| Keep right			| Keep right					 				|
| Speed limit (60km/h)	| Slippery Road (60km/h)						|
| General Caution		| General Caution								|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


| Image			        |     Prediction	        	|%		|  
|:---------------------:|:------------------------------| 		| 
| Stop Sign      		| Stop sign   					|100	|  
| Speed limit (30km/h)	| Speed limit (30km/h)			|100	|  
| Priority road			| Priority road					|100	| 
| Keep right			| Keep right					|100	| 
| Speed limit (60km/h)	| Slippery Road (60km/h)		|100	| 
| General Caution		| General Caution				|100	| 





### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


