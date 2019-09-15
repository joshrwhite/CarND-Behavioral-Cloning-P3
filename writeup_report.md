# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://github.com/white315/CarND-Behavioral-Cloning-P3/blob/master/images/NVIDIA_architecture.jpg "NVIDIA"
[image2]: https://github.com/white315/CarND-Behavioral-Cloning-P3/blob/master/images/original_image.jpg "original"
[image3]: https://github.com/white315/CarND-Behavioral-Cloning-P3/blob/master/images/cropped_image.jpg "cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA's architecture based on the recommendation of Udacity and other classmates. This model can be found in model.py, lines 115-140. The model is pictured below:

![NVIDIA][image1]

The model includes RELU/ELU layers to introduce nonlinearity between each Convultional and Dense layer, and the data is normalized in the model using a Keras lambda layer (code line 116). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 125). 

The model was trained and validated on different data sets to ensure that the model was not overfitting using the data generator function starting at line 40. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer and is driven by mean squared error loss, so the learning rate was not tuned manually (model.py line 142).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In the 'generator()' function, lines 60-69, I add 0.2 to the steering angle measurement if using the left camera and decreasing by 0.2 if using the right camera.

I also cropped the image since the NVIDIA architecture supports an input shape of (66, 320, 3). So in line 117, I use 'model.add(Cropping2D(cropping=((70,25),(0,0))))' to crop 70 pixels from the top and 25 pixels from the bottom.

Original Picture:
![original][image2]

Cropped Picture:
![cropped][image3]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the videos to see if I could replicate what was happening on my own and then seek out other architecture and compiling methods for the best autonomous drive.

After getting a good response from initial attempts, my first step was to use a convolution neural network model similar to the NVIDIA approach because the Udacity team and other classmates havefound luck with this approach. I also added a plethora of ELU activations due to some other suggestions by colleagues.

To combat the overfitting, I modified the model so that I could 1) generate some of my own data and use it in the training process and 2) use the 'model.fit_generator()' command to try out (even if it took longer for each epoch.

The final step was to run the simulator to see how well the car was driving around track one. Even though the steering was shaky, and not as smooth as I'd like it to be, I got a successful lap around the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 115-140) consisted of a convolution neural network with the following layers and layer sizes (see the below visualization of the architecture):

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Image   							| 
| Lambda Normalization	| x / 255.0 - 0.5   							| 
| Cropped         		| 66x320x3 Image   								| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 31x158x24 	|
| ELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 14x77x36 	|
| ELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 5x37x48 	|
| ELU					|												|
| Dropout				| 0.5											|
| Convolution 5x5	    | 3x3 stride, valid padding, outputs 3x35x64 	|
| ELU					|												|
| Convolution 5x5	    | 3x3 stride, valid padding, outputs 1x33x64 	|
| ELU					|												|
| Flatten				| Input of 1x33x64, output of 2112				|
| Dense					| Output of 100									|
| ELU					|												|
| Dense					| Output of 500									|
| ELU					|												|
| Dense					| Output of 10									|
| ELU					|												|
| Dense					| Output of 1									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded three laps on track one going in the counter clockwise direction and then a final lap going in the clockwise direction.

To augment the data sat, I also flipped images and angles so that, for every one captured image, I had six total images to get information from. Angle measurements and images were explained above in Model Architecture & Training (4).

I finally randomly shuffled the data set and put 15% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs I used was 5 since the Udacity videos usually found their way to 5 epochs eventually and I didn't have time for the 'model.fit()' function to go through any more epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Output Video

You can find video of the autonomous lap through this [YouTube link](https://www.youtube.com/watch?v=bJARWhyIkRM&feature=youtu.be)
