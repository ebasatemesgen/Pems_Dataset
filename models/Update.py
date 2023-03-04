#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset





class DatasetSplit(Dataset):
    """
    Splits the datasets by the idxs
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        input, label = self.dataset[self.idxs[item]]
        return input, label


class LocalUpdate(object):
    """
    Model Aggregation (Federated Averaging)
    """
    def __init__(self, args, dataset=None, idxs=None):
        """
        Args:
            args: contains arguments passeds
            dataset: the training dataset 
            idxs: index of the dict users
        """
        self.args = args
        self.loss_func = nn.SmoothL1Loss()                         # It is less sensitive to outliers and prevents exploding gradients 
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch, shuffle=True)
        
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.epochs):
            
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.ldr_train):
                
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()                                   #  Sets the gradients of all its parameters to zero for the local paramaters to learn new values
                log_probs = net(inputs)
                
                # print(inputs.shape)
                # print(log_probs.shape)
                
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                
                ## -------------------------- Prints the steps in each epoch ---------------------------- #
                
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, batch_idx * len(inputs), len(self.ldr_train.dataset),
                            100. * batch_idx / len(self.ldr_train), loss.item()))
                
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

