import torch
import torchvision
import numpy as np
import pandas as pd
from torch import optim
import time


def train(n_epochs, train_loader, valid_loader, model, optimizer, criterion, use_cuda, save_path):
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


def test(test_loader, model, criterion, use_cuda):    
    # initialize lists to monitor correct guesse and total number of data 
    test_loss = 0.0
    class_correct = list(0. for i in range(len(cat_list)))
    class_total = list(0. for i in range(len(cat_list)))

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
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
            
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    # store the class accuracy
    class_accuracy = {i: ((class_correct[i] / class_total[i]) if class_total[i]>0 else 0) 
                      for i in range(len(cat_list))}
    sorted_class_accuracy = sorted(class_accuracy.items(),  key=lambda kv: kv[1], reverse=True)
    
    print('Printing 5 classes with greatest accuracy')
    for i in range(5):
        print('Test Accuracy of %5s: %2d%%' % (
            str(num2cat[sorted_class_accuracy[i][0]]), 100 * sorted_class_accuracy[i][1]))
    
    print('\nPrinting accuracy for each class')
    for i in range(len(cat_list)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                num2cat[sorted_class_accuracy[i][0]], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (num2cat[sorted_class_accuracy[i][0]]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))