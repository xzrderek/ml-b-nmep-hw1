# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`YOUR GITHUB REPO HERE: git@github.com:xzrderek/ml-b-nmep-hw1.git`

## To move to your own repo:

First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```



# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

The nn.module is the foundation for PyTorch neural networks, managing parameters and defining layers. The nn.functional package provides activation and pooling functions used in the forward() method.


## -1.1 What is the difference between a Dataset and a DataLoader?

In PyTorch, a Dataset groups data from various sources, whereas a DataLoader takes a Dataset and creates an iterable with a fixed batch size for processing.


## -1.2 What does `@torch.no_grad()` above a function header do?

The PyTorch function torch.no_grad() disables gradient calculation for a set of code, making it unavailable for backpropagation. This can improve speed.


# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?**

The build.py files act as the intermediate between the configuration files and the working model. They allow other files to use the model and data without worrying about complex configurations or dependencies.

## 0.1 Where would you define a new model?

config/model

## 0.2 How would you add support for a new dataset? What files would you need to change?

The configuration for the model is located in data/build.py and would need to be changed.

## 0.3 Where is the actual training code?

main.py

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)


# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

The build loader function is responsible for creating both the training and validation datasets, as well as their respective data loaders.


### 1.0.1 What functions do you need to implement for a PyTorch Datset? (hint there are 3)

`_getitem_, _len_, _get_transforms`

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

In the initialization of the dataset class, a self.dataset attribute is created and populated with either the CIFAR10 or Medium Image Net dataset.

### 1.1.1 What is `self.train`? What is `self.transform`?

self.train is a bool that tells us if this data is training data or not. in _get_transform. self.transform is a list of image transformation methods.

### 1.1.2 What does `__getitem__` do? What is `index`?

`_getitem_` gets an image from self.dataset with a specified index. Index is the index in the dataset.

### 1.1.3 What does `__len__` do?

_len_ returns the length of the given dataset.

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

self._get_transforms gives a list of image transformations. There's an if statement that controls whether the data is training data or validation data.

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

transforms.Normalize takes in a tensor of an image and normalizes the image. The parameter mean/std sequence of means/stds per channel, while inplace is a boolean parameter indicating inplace normalization.

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

"/data/medium-imagenet/medium-imagenet-nmep-96.hdf5" field stores the data, /data/medium-imagenet/ on honeydew. The weights are over 150 GB

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

Rather than simply checking if the model is trained and returning two different outcomes, the getTransforms function checks whether the model is in training mode and whether data augmentation should be applied. The function also stores the transforms in a list. Additionally, the function divides the data by 256 as a normalization technique.


### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

This `_getitem_`method is different in that it checks if the model is testing. If it is, labels are added, but if not, the labels become arbitrary. Additionally, because these images are composed of pixel values, they are split, whereas the CIFAR10 dataset was not split.

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.


# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

lenet and resnet

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

init, forward, they inherit form nn.Module

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

5 layers, 2 CNN, 3 linear, 0.099276 M



# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

lenet_base.yaml - cifar10, lenet; resnet18_base.yaml - cifar10, resnet18, resnet18_medium_imagenet.yaml - resnet18, medium_imagenet

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

The main function in main.py takes a configuration file as input and trains, prints, and saves a model based on the provided configuration. The function begins by creating the training and validation datasets and their respective data loaders. The model is then built based on the provided configuration, and the relevant information is logged. The optimizer and criterion are set, and the model is trained using the training data. After training, the predictions are saved to a CSV file in Kaggle format.

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

The validate function checks the model accuracy during one epoch without performing backpropagation, and prints the weights in the terminal. It compares the current model state predictions to the correct outputs in the validation dataset. The evaluate() method, on the other hand, returns a prediction array. Unlike the validate() method, evaluate() is used for making predictions for the real use of the model, rather than merely evaluating and displaying the model accuracy.


# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activatioons after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)

Lenet has 0.099276 M parameters and 1.1 GB memory. AlexNet has 57.82324 M parameters and 2.4 GB memory.

## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

`77.3 % accuracy`



# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

Graphed in weights and biases

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

https://api.wandb.ai/links/ml-b/bz82vffv
https://api.wandb.ai/links/ml-b/sh29kiy0
https://api.wandb.ai/links/ml-b/yif5s1e2
https://api.wandb.ai/links/ml-b/y7xg2uta

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

https://api.wandb.ai/links/ml-b/xp8uj9ex
https://api.wandb.ai/links/ml-b/nkr3oz92
https://api.wandb.ai/links/ml-b/4wfvsmh6
https://api.wandb.ai/links/ml-b/hqflxowv

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

https://api.wandb.ai/links/ml-b/hy45dp1v
https://api.wandb.ai/links/ml-b/ysvyiwib
https://api.wandb.ai/links/ml-b/18p7uhxo
https://api.wandb.ai/links/ml-b/de1e0z4h

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 

https://api.wandb.ai/links/ml-b/gm1c1hw4 

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.


https://api.wandb.ai/links/ml-b/ei3khtbl
https://api.wandb.ai/links/ml-b/6jvr0i1d
https://api.wandb.ai/links/ml-b/pfw8efjg
https://api.wandb.ai/links/ml-b/ezw9aeo0


## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`



# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/models.py`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

https://api.wandb.ai/links/ml-b/5cqtao31
https://api.wandb.ai/links/ml-b/6fv4csoi
https://api.wandb.ai/links/ml-b/bl1js0mt
https://api.wandb.ai/links/ml-b/3rg3yz2y

## 6.1 Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

Check visualize.ipynb in home directory.

# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot). The competition ends on __TBD__.

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉

Attempted running different batch sizes over varying epochs. Best for me was deafult batch size with 10 epochs. Attempted 40 epochs (was worse, not really sure why), 4x larger LR and batch size and 10 epochs, and 8x larger LR and batch size and 10 epochs.