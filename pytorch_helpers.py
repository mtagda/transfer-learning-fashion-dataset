import torch
import torchvision
import numpy as np
import pandas as pd
from torch import optim
import time


def train(n_epochs, train_loader, valid_loader, model, optimizer, criterion, use_cuda, save_path, scheduler=False):
    """returns trained model"""
    since = time.time()
    
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    # initialize lists of values of train and valid losses over the training process
    train_loss_history = []
    valid_loss_history = []
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        
        # Call the learning rate scheduler if given
        if scheduler:
            scheduler.step()
            
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # record the average training loss and add it to history
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        train_loss_history.append(train_loss)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # update the average validation loss and add it to history
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        valid_loss_history.append(valid_loss)
            
        # Print training/validation statistics 
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            n_epochs,
            train_loss,
            valid_loss
            ))
        
        # Save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
            
    # Print training time        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # return trained model
    return model, train_loss_history, valid_loss_history


def correct_top_k(output, target, topk=(1,)):
    """Returns a tensor with 1 if target in top-k best guesses and 0 otherwise"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).sum(0, keepdim=True)
        return correct[0]
    
    
def test(test_loader, model, criterion, cat_lookup, use_cuda):    
    # initialize lists to monitor correct guesse and total number of data 
    test_loss = 0.0
    class_correct = list(0. for i in range(len(cat_lookup)))
    class_correct_top_5 = list(0. for i in range(len(cat_lookup)))
    class_total = list(0. for i in range(len(cat_lookup)))
    model.eval() # prep model for evaluation

    for data, target in test_loader:
        if use_cuda:
            data, target, model = data.cuda(), target.cuda(), model.cuda()
        #else:
        #    model = model.cpu()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        
        # get 1 if target in top-1 predictions
        correct_top_1 = correct_top_k(output, target, topk=(1,))
        # get 1 if target in top-5 predictions
        correct_top_5 = correct_top_k(output, target, topk=(5,))
        
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct_top_1[i].item()
            class_correct_top_5[label] += correct_top_5[i].item()
            class_total[label] += 1
            
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    
    class_accuracy_top_1 = {}
    print('\nPrinting accuracy for each class')
    for i in range(len(cat_lookup)):
        if class_total[i] > 0:
            accuracy_top_1 = 100 * class_correct[i] / class_total[i]
            accuracy_top_5 = 100 * class_correct_top_5[i] / class_total[i]
            class_accuracy_top_1[i] = accuracy_top_1
            print('Test accuracy of %5s: \nTop-1 accuracy: %2d%% (%2d/%2d) \nTop-5 accuracy: %2d%% (%2d/%2d)'.format(
            ) % (
                cat_lookup[i], accuracy_top_1,
                np.sum(class_correct[i]), np.sum(class_total[i]),
                 accuracy_top_5, np.sum(class_correct_top_5[i]), np.sum(class_total[i])))
            
            
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (cat_lookup[i]))

    print('\nPrinting 5 classes with greatest top-1 accuracy')
    sorted_class_accuracy = sorted(class_accuracy_top_1.items(),  key=lambda kv: kv[1], reverse=True)
    for i in range(5):
        print('Test Accuracy of %5s: %2d%%' % (
            str(cat_lookup[sorted_class_accuracy[i][0]]), sorted_class_accuracy[i][1]))
        
    print('\nTest Accuracy (Overall): \nTop-1 accuracy: %2d%% (%2d/%2d) \nTop-5 accuracy: %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total),
         100. * np.sum(class_correct_top_5) / np.sum(class_total),
        np.sum(class_correct_top_5), np.sum(class_total)))
    
