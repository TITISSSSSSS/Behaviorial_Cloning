# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Read and explore dataset provide by Udacity 
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/raw_image.png "raw_image"
[image2]: ./examples/steering.png "steering"
[image3]: ./examples/crop.png "crop Image"
[image4]: ./examples/bir.png "bir Image"
[image5]: ./examples/flip.png "flip Image"
[image6]: ./examples/model.png "model Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
## Read and explore dataset
I use the dataset provided by Udacity which is enough to train my model, so i did not record images myself. 

If you want, you can use .[Udacity simulator](https://github.com/udacity/self-driving-car-sim) to generate your own dataset.

The images output from simulator is 160x320x3 dimensions. I randomly print out three images with three different positions.

![alt text][image1]

The dataset contains 8036 images. One pontential issue of this dataset is that most of the steering angles are close to zero.

![alt text][image2]

---
## Preprocess and augment dataset
### Data Preprecessing 
**Cropping ane resizing images**
The top of images mostly capture sky and tress and hills and other elements that might be more distracting the model. Beside  bottom portion of the image captures the hood of the car.

In order to focus on only the portion of the image that is useful for predicting a steering angle, i crop 55 pixels from the top and 25 pixels from the bottom.

![alt text][image3]
**Normalizing the data and mean centering the data**
* pixel_normalized = pixel / 127.5
* pixel_mean_centered = pixel / 127.5 - 1

### Data Augmentation 
**Adjust the brightness of the images**
![alt text][image4]
**Flipping Images**

Flipping images and taking the opposite sign of the steering measurement
![alt text][image5]
**Using multiple cameras**

The simulator can capture three position images which are a center, right and left camera image. I can use these side camera images to increase the dataset. More importantly, it will be helpful to recover the car from being off the center.

In order to use these side cameras, I will add a small correction factor to these cameras. 
* For the left camera: center steering angle + 0.25
* For the right camera: center steering angle - 0.25 
---
## Build a convolution neural network to predicts steering angles
I implmented the end-to-end CNN model to predict the steering angles which is based on .[NVIDIA architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

**My final model consisted of the following layers:**
* Optimizer: Adam

* Loss: Mean square error

* Batch size: 32

* Epoch: 3 
![alt text][image6]
---
## Results

* .[Behavioral_Cloning.py](Behavioral_Cloning.py): project file

* .[drive.py](drive.py): drive a car in autonomous mode

* .[model.h5](model.h5): trained model to predict steering angles

* .[video.py](video.py): create video file

* .[run1.mp4](run1.mp4): a final result, drive the car on the track 1
---
## Potential shortcomings and future works
My model can drive car safely on the tack1, but the speed is limited. When the speed gets faster, the car will shake a bit from side to side. 

### solutions:
* Since the dataset is unbalance and all the steering angle are cloesed to zero, i can try to balance this dataset which will help to make the car stable.

* Try to apply binary and color threshold techniques to preprocess the data captured by cameras. It will be helpful when the car drives under shadows.