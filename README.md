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


The dropout layer avoids overfitting as this model has been trained for 100 epochs.

You can find the trained model file [here](https://github.com/chinmaykumar06/cats-vs-dogs-classifier/blob/master/models/augmented_cnn.h5).

For this model the results obtained were:
* **Validation accuracy: 81.25%**
* **Training accuracy: 82.50%**

### 3. Transfer Learning using VGG16


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

## Model Design and Implementation

Our problem statement was to classify colored images into two categories viz Cats and Dogs, for extracting features from an image CNNs are the best solution, although they often require hefty computations. Hence I started with a simple model consisting of 4 Convolutional layers preceeded by MaxPool layers, for introducing non linearity in the model the activation function used throughtout was ReLU. I trained this model for 20 epochs and acheived a validation accuracy of abouit 73% which is pretty good for a simple model as this, however greater accuracy could have been used if I had trained it on a larger dataset. Hence in my next model I peformed data augmentation!

In my second model I performed various operations on the training images like shearing, shifting and rotation; thereafter those images were feeded to the model, I kept the architecture same as the previous model, however this time I trained the model for 100 epochs hence added a dropout layer to avoid overfitting. This model gave me a validation accuracy of 81.25%. Note that I performed data augmentation using the ImageDataGenerator provided by keras hence the size of my dataset didn't increase. This Data generator just performs the specified operations on the images randomly in every batch before inputing it to the model, hence making our data more diverse. The # images used for training is still the same. Getting better results using less data is a great acheivement since we can do better predictions using the same computational power!!

Finally I went for Transfer learning using the popular [VGG16](https://arxiv.org/pdf/1409.1556.pdf) architecture. Unlike my last models this model had many many parameters as I kept the VGG16 layer as functional and didn't freeze it while training! I trained the model for 30 epochs. Since this was a complex architecture I used [callbacks](https://github.com/chinmaykumar06/cats-vs-dogs-classifier/tree/master/Callbacks) to create checkpoints at the end of each epoch so that I could revert back if my training stopped abruptly, also this history of epochs can be used for further fine tuning the model. For all the models I have used RMSprop as an optimizer, in this model I used a particularly low learning rate of 1e-5 for obtaining best results. After the tarining I got a validation accuracy of 96% which according to me is a great result for just 2000 training images. 

## Testing, Validation and Analysis

I tested all three models using my test dataset and the images taken from internet. The testing accuracy for the three models were:

* My Model: 72.29%
* My Model with Data Augmentation: 80.9%
* Transfer Learning using VGG16: 95.8%

The testing done on the images downloaded from the net also gave great insight about the model performance. One of the images downlaoded from the net consisted of many dogs and cats as shown below:

<img src="https://github.com/chinmaykumar06/cats-vs-dogs-classifier/blob/master/images_from_net/confuse.jpg" width="300">

Statistically this image has more cats than dogs, when the image was given as input to the models it was predicted as a dog with high confidence in all three models, which is not surprising as the dogs occupy a larger part of the image as compared to the cats and hence when the convolutional layers extract the features from the image the features of dogs are in majority.

To correctly observe the working of convolutional layers and maxpool layers I have even observed the output of the individual Conv layers and MaxPool Layers for the first model, it gave me a great insight into the process of filtering and pooling. 

## Adding new train dataset:

If you want to add a new training image to the available datasets, you can add the images of cats and dogs to **/data/train/cats** and **/data/train/dogs** respectively. The image size doesn't matter as it it resized before training in the code.

## Implement this project

### I. If you have a dedicated GPU then follow these steps to run this model using [miniconda](https://conda.io/en/latest/):

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

**Clone** the repository. 

(You will require Git for this)

```sh
git clone https://github.com/chinmaykumar06/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```

In Windows: After you have cloned the repository in the same folder open the Command Prompt or else the Anaconda Prompt, for Command Promt run activate to activate your base environment.

**Create** my_project_env.  Running this command will create a new `conda` environment named my_project_env.
```
conda create --name my_project_env python =3.7
```

**Download** the dependencies.  Running this command will download the dependencies and libraries in the created environment.

```
pip install -r requirements.txt
````

**Verify** that the environment was created in your environments:

```sh
conda info --envs
```
**If you think some error happend while performing the above process you can clean up the libraries and uninstall athe environment and try again!**

**Cleanup** downloaded libraries (remove tarballs, zip files, etc):

```sh
conda clean -tp
```

**Uninstalling**

To uninstall the environment:

```sh
conda remove --name my_project_env --all
```

**Once you are sure that your environment is installed properly,**

**Activate** the enoviornment.  Running this command will activate the created environment where you can install the dependencies.

`````
conda activate my_project_env
`````

**Open** jupyter notebook.  Running this command will open jupyter notebook.

```
jupyter notebook
````

**Open the CatsVsDogs.ipynb** and make sure you choose the right environment where you installed your libraries and then start executing the cells!

To exit the environment when you have completed your work session, simply close the terminal window.

### II. If you want to run this project on you machine with a CPU, roughly these are the dependencies required:
* Jupyter notebook
* Tensorflow = 2.1
* Keras = 2.4.3
* Python 3.7
* Matplotlib
* Seaborn
* Scikit-Learn
* Pandas
* Numpy
* pydotplus
* graphviz

### III. Using Google Colab
This is highly recommended as Colab provides a GPU runtime and you don't have to install the dependencies on your system. Also this is the least complex process!

1. Open https://colab.research.google.com, click **Sign in** in the upper right corner, use your Google credentials to sign in.
2. Click **GITHUB** tab, paste https://github.com/chinmaykumar06/cats-vs-dogs-classifier.git and press Enter
3. This repository will get cloned in your Drive in the folder Colab Notebooks
4. Go to the directory and open CatsVsDogs.ipynb using Colab.
5. Click **Runtime -> Change runtime type** and select **GPU** in Hardware accelerator box.
6. Now mount your drive and using cd change your path to the cloned repository so that you can use the data and images without adding complex path names.
7. You can now start executing the codeblocks!
