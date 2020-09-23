# Cats Vs Dogs Classifier
## To view the Project Notebook
Since I have implemented three architectures in the same notebook, the size of the ipynb notebook is bigger than usual and hence it may not open in GitHub. You can view the Project notebook [here](https://nbviewer.jupyter.org/github/chinmaykumar06/cats-vs-dogs-classifier/blob/master/CatsVsDogs.ipynb) .

## Overview of the Project
<img src="https://github.com/chinmaykumar06/cats-vs-dogs-classifier/blob/master/output.png" width="1200">

This project is inspired by the Kaggle Challange held in 2003 in which thousands of images of cats and dogs were given and a model was to be built to classify those images into cats and dogs. The best accuracy achieved in that competition was 98%!

I used a subset of that data and built my model, in the original dataset there were around 25000 images for training but I am only using 2000 images..
I used three different achitectures to train this dataset and increased the validation accuracy from around 73% to 96%!!!

The basic steps follwed in each architecture was:
* Resize the images to desired input size and apply necesaary pre processing like shear,rotate,shift etc.
* Design, train and validate the model using the dataset.
* Download an image from the internet and test it on the trained model.

## Model Architectures
### 1. My Model
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 150x150x3 RGB image   						| 
| Convolution 3x3     	|  filters = 32 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Convolution 3x3     	|  filters= 64 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Convolution 3x3     	|  filters = 128 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Convolution 3x3     	|  filters = 128 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Flatten				|												|
| Fully connected(Dense)		| Outputs 512									|
| RELU					|												|
| Output				| Outputs 1, activation:sigmoid 								|

You can find the trained model file [here](https://github.com/chinmaykumar06/cats-vs-dogs-classifier/blob/master/models/my_model.h5).

For this model the results obtained were:
* **Validation accuracy:72.8%**
* **Training accuracy:92.55%**

### 2. Data Augmentation
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 150x150x3 RGB image   						| 
| Convolution 3x3     	|  filters = 32 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Convolution 3x3     	|  filters= 64 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Convolution 3x3     	|  filters = 128 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Convolution 3x3     	|  filters = 128 						|
| RELU					|												|
| MaxPooling 2x2     	|  						|
| Flatten				|												|
| Dropout				| Probability 50%								|
| Fully connected(Dense)		| Outputs 512									|
| RELU					|												|
| Output				| Outputs 1, activation:sigmoid 								|

You can find the trained model file [here](https://github.com/chinmaykumar06/cats-vs-dogs-classifier/blob/master/models/augmented_cnn.h5).

For this model the results obtained were:
* **Validation accuracy:72.8%**
* **Training accuracy:92.55%**

#### 3. Transfer Learning using VGG16


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 150x150x3 RGB image   						| 
| VGG16    	|  Functional 						|
| Flatten				|												|
| Fully connected(Dense)		| Outputs 256								|
| RELU					|												|
| Output				| Outputs 1, activation:sigmoid 								|

You can find the trained model file [here](https://github.com/chinmaykumar06/cats-vs-dogs-classifier/blob/master/models/vgg16_cnn.h5).

For this model the results obtained were:
* **Validation accuracy:96.1%**
* **Training accuracy:98.65%**
