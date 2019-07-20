import os
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler


class FIDataset(Dataset):
    """ Fashion Image Dataset
    """
    
    def __init__(self, dir, dataframe, transform, cat_lookup):
        super(FIDataset, self).__init__()
        self.dataframe = dataframe
        self.dir = dir
        self.transform = transform
        self.cat_lookup = cat_lookup
        
    def __getitem__(self, idx):
        line = self.dataframe.iloc[idx]
        cat = line.articleType
        cat_id = self.cat_lookup[cat]
        img_path = os.path.join(self.dir, str(line.id)+'.jpg')
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, cat_id
            
    def __len__(self):
        return len(self.dataframe)


def split_train_valid(train_data, valid_size):
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    return train_sampler, valid_sampler


def imshow(img):
    # helper function to un-normalize and display an image
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    
    
def plot_sample_data(dataloader, num, cat_lookup):
    # obtain one batch of training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    # convert to numpy for display
    images = images.numpy() 
    labels = labels.numpy() 
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display images
    for idx in np.arange(num):
        ax = fig.add_subplot(1,5, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(cat_lookup[labels[idx]])
        
        
def plot_sample_data_model(dataloader, num, model, cat_lookup, use_cuda=True):
    import torch
    # obtain one batch of training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    # convert to numpy for display
    labels = labels.numpy() 
    
    # move model inputs to cuda, if GPU available
    if use_cuda:
        images = images.cuda()
    else:
        model = model.cpu()

    # get sample outputs
    output = model(images)
    
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())
    
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(15, 6))
    # display images
    for idx in np.arange(num):
        ax = fig.add_subplot(2,num/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx].cpu())
        ax.set_title("{} ({})".format(cat_lookup[preds[idx]], cat_lookup[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
        
        
def plot_training_and_valid_loss(train_loss_history, valid_loss_history):
    # Define a helper function for plotting training and valid loss
    plt.title("Training and Validation Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,n_epochs+1),train_loss_history,label="Training Loss")
    plt.plot(range(1,n_epochs+1),valid_loss_history,label="Validation Loss")
    plt.xticks(np.arange(1, n_epochs+1, 1.0))
    plt.legend()
    plt.show()
    