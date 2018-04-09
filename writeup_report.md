# **Behavioral Cloning** 

## Writeup Project 3 - Self Driving Car Nanodegree

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Images/left_2017_07_01_20_02_32_143.jpg "Left Image"
[image2]: ./Images/center_2017_07_01_20_02_32_143.jpg "Center Image"
[image3]: ./Images/right_2017_07_01_20_02_32_143.jpg "Right Image"
[image4]: ./Images/LenetArchitecture.png "LeNet Architecture"
[video1]: ./run1.mp4 "Output Video"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup\_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the following line in my command window:
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was shaped using the LeNet neural network architecture as a base:
![alt text][image4]

Other layers were added to the LeNet Architecture to tune it for this specific project. It consists of 4 convolutional layers and REctified Linear Units (RELUs) as the activation functions to introduce nonlinearity (model.py lines 55, 57, 59, and 61). Max Pooling layers were used after each convolutional layer to down-sample, reduce the samples dimensions, and prevent overfitting (model.py lines 56, 58, 60, 62). The whole neural network is flattened (model.py line 63) after the last convolutional layer and max pooling layer. The network continues with 3 fully connected layers (or dense layers) to output arrays of sizes starting at 120 and ending with the final array of size 1 (lines 64, 66, and 68). Dropout was also implemented twice in between the first and second dense layer (model.py line 65) and in between the second and third dense layers (model.py line 67). A .8 was used for each dropout layer which means that 80% of the neurons would be ignored during training to prevent "memorization" or over fitting of the simulation track and to be able to generalize better for different scenarios. 

It is important to also mention the use of the Lambda layer to begin the training process to be able to normalize the entire data set.

My final model consisted of the following layers:

| Layer						|
|:-------------------------:| 
| Lambda					|
| cropping					|
| Convolution 5x5 + Relu	|
| Max pooling				|
| Convolution 5x5 + Relu	|
| Max pooling				|
| Convolution 3x3 + Relu	|
| Max pooling				|
| Convolution 3x3 + Relu	|
| Max pooling				|
| FLATTEN					|
| Dense (120)				|
| Dropout					|
| Dense (84)				|
| Dropout					|
| Dense (1)					|

 
#### 2. Attempts to reduce overfitting in the model

As mentioned above, the model contains a total of two dropout layers in between Dense layers in order to reduce overfitting (model.py lines 65 and 67). 

In line 71 of model.py we compile the neural network using a mean squared error and an adam optimazer to calculate loss and automatically assign a learning rate for the network. Then, in line 72 of model.py, the model was trained and validated in a sample data set that was split from the whole data set. By splitting 20% of the data set to validate the model it allowed us to check for overfitting if the difference between the loss and the validation loss was too great. 

The model was saved to an .h5 file in line 74 of the model.py file. This file was used to run the model in autonomous mode using the simulator provided by Udacity. The model landed succesful results by allowing the car to continuously drive around the center of the track without leaving the drivable portion of the same.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

At first I tried gathering my own data. I recorded the car while driving around the track in simulation mode for a few laps. Then, I turn the car around to gather data going to oposite way. I augmented the data in the code but still, after training the model, could not get the car to drive past the first curve.

Then I decided to try using the data made available to students by Udacity. This data consisted of over 20,000 images counting the center, left, and right camera images. The model seemed to train better when using those so I decided to continue improving the model using such data. Instead of tunning my model by recording recovery laps using the simulator, I decided to implement the steering angle correction method. I used a steering correction value of .3 (model.py line 35), added that correction angle to the left camera images and subtracted that correction angle to the right camera images. Using this method I immediately saw substantial improvement. The car was able to drive around the track but did go over some yellow lines at times. I decided to continue improving the model by gathering and recording more data in the section of the track in which the car would drive over the yellow line, and in the curve right after the brige where the right side of the track does not display a yellow line but just a dirt patch. I gathered around 4,000 additional images and retrained the model, yielding successful results of a car going around the track in autonomous mode always staying well inside the yellow lines or towards the middle of the road.

The images below are examples of additional images gathered by me in simulation mode:

Center image:
![alt text][image1]
Left image:
![alt text][image2]
Right image:
![alt text][image3]

To see the video of my car driving autonomously, follow [this link](./run1.mp4)

To record the video above I use the following line of code in my command window to gather the images frame by frame and save them in a folder called run1:
```sh
python drive.py model.h5 run1
```

Then I used the following line of code to create the actual .mp4 file using those frames previously saved in the run1 folder:
```sh
python video.py run1
```

#### 5. Final Thoughts

The process of gathering data, loading it into the AWS instance using command windows, unzipping the images, running the code and training the neural network was very time consuming. A lot of trial and error and experimentation was necessary. There are many parameters that can be tuned and many different ways to accomplish the same goal.

Despite being pretty sattisfied with my final result, I do believe that using a generator could be a better way to approach this project and make it more scalable to other circumstances, since it would allow to train a lot more data without facing the issue of running out of memory.

Also, it is important to make sure that the images are read as RGB images and not BGR since the drive.py script used to train is expecting RGB images and using BGR can yield less than ideal results.

Overall a great project, definitely a hard project, but one to learn a whole lot from.