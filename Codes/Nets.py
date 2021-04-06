# this file contains various datasets architectures that can later be imported and used in DeepEvaluator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Linear, Sequential, ReLU, Conv1d, MaxPool1d, BatchNorm1d, Module, CrossEntropyLoss, MSELoss
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

from DataExtractor import DataExtractor

import matplotlib.pyplot as plt

import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.uniform_(m.bias.data)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.uniform_(m.bias.data)

class FC4(nn.Module):
        def __init__(self, dropout_rate):
            super(FC4, self).__init__()


            self.fcModel = Sequential(
                nn.Linear(350,2048),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(2048,1024),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(1024,512),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(512,256),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(256,128),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(128,2),
                nn.Dropout(dropout_rate),

            )

            #self.fcModel.apply(weights_init)

        def forward(self, x):

            xflat = x.view(x.size(0), -1)
            x = self.fcModel(xflat)

            # Apply softmax to x
            return x

class CNN2(nn.Module):
        def __init__(self, dropout_rate):
            super(CNN2, self).__init__()


            self.cnnModel = Sequential(
                # First layer
                nn.Conv1d(1, 10, kernel_size = 5),
                # nn.BatchNorm1d(10),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),
                MaxPool1d(kernel_size=3, stride=2),

            )

            self.fcModel = Sequential(
                nn.Linear(1720,2048),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(2048,1024),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(1024,512),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(512,256),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(256,128),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(128,2),
                nn.Dropout(dropout_rate),

            )

            # self.cnnModel.apply(weights_init)
            # self.fcModel.apply(weights_init)

        def forward(self, x):
            # Pass data through conv1

            x = self.cnnModel(x)
            xflat = x.view(x.size(0), -1)
            x = self.fcModel(xflat)

            # Apply softmax to x
            return x


class CNN8(nn.Module):
        def __init__(self, dropout_rate):
            super(CNN8, self).__init__()


            self.cnnModel = Sequential(
                # First layer
                nn.Conv1d(1, 10, kernel_size = 30),
                # nn.BatchNorm1d(10),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                # Second layer
                nn.Conv1d(10, 10, kernel_size = 10),
                # nn.BatchNorm1d(10),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),



            )

            self.fcModel = Sequential(
                nn.Linear(3120,2048),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(2048,1024),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(1024,512),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(512,256),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(256,128),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(128,2),
                nn.Dropout(dropout_rate),

            )

            self.cnnModel.apply(weights_init)
            self.fcModel.apply(weights_init)

        def forward(self, x):
            # Pass data through conv1

            x = self.cnnModel(x)
            xflat = x.view(x.size(0), -1)
            x = self.fcModel(xflat)

            # Apply softmax to x
            return x


class CNN9(nn.Module):
        def __init__(self, dropout_rate):
            super(CNN9, self).__init__()


            self.cnnModel = Sequential(
                # First layer
                nn.Conv1d(1, 10, kernel_size = 5),
                # nn.BatchNorm1d(10),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                # Second layer
                nn.Conv1d(10, 10, kernel_size = 10),
                # nn.BatchNorm1d(10),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                # Third layer
                nn.Conv1d(10, 10, kernel_size = 30),
                # nn.BatchNorm1d(10),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),



            )

            self.fcModel = Sequential(
                nn.Linear(30300,2048),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(2048,1024),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(1024,512),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(512,256),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(256,128),
                nn.Dropout(dropout_rate),
                ReLU(inplace=True),

                nn.Linear(128,10),
                nn.Dropout(dropout_rate),

            )

            self.cnnModel.apply(weights_init)
            self.fcModel.apply(weights_init)

        def forward(self, x):
            # Pass data through conv1

            x = self.cnnModel(x)
            xflat = x.view(x.size(0), -1)
            x = self.fcModel(xflat)

            # Apply softmax to x
            return x
