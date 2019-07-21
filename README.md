# transfer-learning-fashion-dataset
Applying Few shot / Transfer Learning on Fashion Product Images Dataset​ from Kaggle

## Getting started: downloading the data
The best way to download the dataset is to use the kaggle API (see https://github.com/Kaggle/kaggle-api#datasets). Then, you can simply download the .zip file using the following command:

```{r, engine='bash', count_lines}
kaggle competitions download -c fashion-product-images-dataset
```

Finally, you should unzip fashion-product-images-dataset.zip in `DATASET_PATH`.

## Goal:
The goal is to train a classifier on the 142 different `articleType` classes.

## Distribution of the data
The following charts shows how the data among the 142 categories is distributed. 
<img src="images/distribution.png"> 

## Using weighting to handle the imbalanced classes 

Reference: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
Weight balancing balances our data by altering the weight that each training example carries when computing the loss. Normally, each example and class in our loss function will carry equal weight i.e 1.0. But sometimes we might want certain classes or certain training examples to hold more weight if they are more important. That's what we do in the next step: we assign to each class i the following weight w_i:

<img src="images/weight.png"> 

## Create master train and test splits of the valid image data
Use everything in even years as the training set, and everything in an odd year as the test split.
```{r, engine='python', count_lines}
training_data = styles[styles['year'].astype('int') % 2 == 0]
testing_data = styles[styles['year'].astype('int') % 2 == 1]
```

## Sample data from the training dataset

<img src="images/sampledata.png"> 
<img src="images/sampledata2.png"> 

## Train a classifier using transfer learning
Initializing from ResNet50 trained on ImageNet. ResNet50 is a deep residual network and a very good architecture with 50 layers perfectly suitable for image classification problems.

In the following, we will consider 3 approaches for this problem.

### Approach 1 
See FIDataset_transfer_learning_FConly.ipynb

We freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.

```{r, engine='python', count_lines}
# Specify model architecture 
model = models.resnet50(pretrained=True)

# Freeze training for all "features" layers
for param in model.parameters():
    param.requires_grad = False
    
# To reshape the network, we reinitialize the classifier’s linear layer
n_inp = model.fc.in_features
model.fc = nn.Linear(n_inp, len(cat_list))
```
We train the network for 20 epochs. The figure below shows train and valid loss during training.

<img src="images/loss_fconly.png"> 

The overall **test accuracy is 49%** (10243/20634). Below we can see sample images along with predicted and true labels.

<img src="images/sample_resuls_fconly.png"> 

The top 5 classes with greatest accuracy are:

```
Test Accuracy of Sunglasses: 100%
Test Accuracy of Water Bottle: 100%
Test Accuracy of Footballs: 100%
Test Accuracy of  Ties: 97%
Test Accuracy of Backpacks: 94%
```

