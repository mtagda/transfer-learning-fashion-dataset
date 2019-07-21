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

### Approach 2
See FIDataset_transfer_learning_direct.ipynb

We replace the last fully connected layer with a new one with 142 outputs and retrain the weights of the **whole** network.

```{r, engine='python', count_lines}
# Specify model architecture 
model = models.resnet50(pretrained=True)

# To reshape the network, we reinitialize the classifier’s linear layer
n_inp = model.fc.in_features
model.fc = nn.Linear(n_inp, len(cat_list))
```
We train the network for 20 epochs. The figure below shows train and valid loss during training.

<img src="images/loss_direct.png"> 

The overall **test accuracy is 68%** (14144/20634). Below we can see sample images along with predicted and true labels.

<img src="images/sample_resuls_direct.png"> 

The top 5 classes with greatest accuracy are:

```
Test Accuracy of Sunglasses: 100%
Test Accuracy of Kurta Sets: 100%
Test Accuracy of Earrings: 100%
Test Accuracy of Accessory Gift Set: 100%
Test Accuracy of Water Bottle: 100%
```

### Approach 3
See FIDataset_transfer_learning_2steps.ipynb

In this approach, we take 2 steps to train the classifier. First, we train the full network (all layers) on the top 20 classes (20 outputs). Then we replacing the 20-class FC layer with a 141-class output layer, and fine tune on the rare classes.

#### Step 1: Train the full network (all layers) on the top 20 classes
Retrain the weights of the whole network.

```{r, engine='python', count_lines}
# Specify model architecture 
model_top20 = models.resnet50(pretrained=True)

# To reshape the network, we reinitialize the classifier’s linear layer
n_inp = model_top20.fc.in_features
model_top20.fc = nn.Linear(n_inp, len(top_classes_names))
```
We train the network for 20 epochs. The figure below shows train and valid loss during training.

<img src="images/loss_20.png"> 

The overall **test accuracy is 87%** (13243/15143). Below we can see sample images along with predicted and true labels.

<img src="images/sample_resuls_20.png"> 

The top 5 classes with greatest accuracy are:

```
Test Accuracy of Watches: 100%
Test Accuracy of Sunglasses: 100%
Test Accuracy of Jeans: 99%
Test Accuracy of Shirts: 99%
Test Accuracy of Belts: 98%
```

#### Step 2: Fine tuning
Replace the 20-class FC layer with a 142-class output layer. Freeze the initial 5 layers of the pretrained model (model_top20) and train just the remaining layers again. The top layers would then be customized to the new data set. Since the new data set contains 122 new classes with low similarity, it is a good idea to retrain and customize the higher layers according to the new dataset. The initial layers are kept pretrained and the weights for those layers are frozen.

```{r, engine='python', count_lines}
# Freeze the first 5 layers of the model
layers_to_freeze=5
layer_count=0

for child in model_142.children():
    layer_count+=1
    if layer_count <= layers_to_freeze:
        for param in child.parameters():
            param.requires_grad = False
            
# To reshape the network, we reinitialize the classifier’s linear layer
n_inp = model_142.fc.in_features
model_142.fc = nn.Linear(n_inp, len(uniquie_article_types))
```
We train the network for 20 epochs. The figure below shows train and valid loss during training.

<img src="images/loss_142.png"> 

The overall **test accuracy is 70%** (14640/20634). Below we can see sample images along with predicted and true labels.

<img src="images/sample_resuls_142.png"> 

The top 5 classes with greatest accuracy are:

```
Test Accuracy of Sunglasses: 100%
Test Accuracy of Earrings: 100%
Test Accuracy of Accessory Gift Set: 100%
Test Accuracy of Rain Jacket: 100%
Test Accuracy of Water Bottle: 100%
```

