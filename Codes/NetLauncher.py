# this function trains a NN model and prints intermediates results.
# ARGUMENTS
# num_epochs : number of epochs the model is trained on
# batch size : size of each minibatch used.
# learning_rate : learning rate used
# dropout_rate : dropout rate
# weighted : boolean indicating wether or not the various classes the model tries to classify are being weighted when computing  the loss
# model_name : name of the specific model architecture one wants to build. The corresponding achitecture must be implemented in the Net.py file
# X/Y_path :  X and Y training set paths
# validX/Y_path : x and y validation set path
# testX/Y_path : X and y testing set paths
# binary class : boolean indicating wether the model predicts binary classes or not

# OUTPUT : a model being trained on the referenced training set and saved in the
# Models folder. At the end of each epoch, if the current version of the model is performing better on the
# validation set than the previous best model version, then the current version will be saved as a checkpoint.
# furthermore, the plot of the evolution of the validation and training loss will be saved in the Loss folder.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Linear, Sequential, ReLU, Conv2d, MaxPool2d, BatchNorm2d, Module, CrossEntropyLoss, MSELoss
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

from DataExtractor import DataExtractor
from sklearn.metrics import classification_report

from DeepEvaluator import  DeepEvaluator
from PerformanceEvaluator import PerfEvaluator

import matplotlib.pyplot as plt
import math
import os.path
from tqdm import tqdm
import numpy as np


def netLauncher(num_epochs, batch_size, learning_rate, dropout_rate, weighted, model_name, train_path , valid_path,
                test_path, binary_class):

    # make data go through the net
    # random_data = torch.rand((10, 1, 310))
    # my_nn = Net()
    # result = my_nn(random_data)
    # # print (result)

    # deep learning model
    deep_evaluator = DeepEvaluator(num_epochs, learning_rate, dropout_rate, weighted, model_name, binary_class)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # training set
    train_data = deep_evaluator.loadDataset(train_path, batch_size, "training", binary_class)

    # buidling validation set ( dans mon cas j'avais plusieurs files, à voir ici) HACKATON VALIDATION PATH
    # valid_data = []
    # for i in range(95):
    #     valid_f_X_path = validX_path.format(i)
    #     valid_f_Y_path = validY_path.format(i)
    #     if os.path.isfile(valid_f_X_path):
    #         valid_data += [deep_evaluator.loadDataset(valid_f_X_path, valid_f_Y_path, batch_size, "validation", binary_class)]

    valid_data = deep_evaluator.loadDataset(valid_path, batch_size, "validation", binary_class)# HACKATON valid data

    # IDs of both model and dataset used for saving trained model under unique name
    dataset_ID = train_path.split('/') # HACKATHON : mettre ici un système pour différencier les datasets
    dataset_ID =  dataset_ID[0]
    model_ID = model_name +"_" +str(batch_size) + "_" + str(dropout_rate) + "_" + str(weighted) + "_" + str(learning_rate) + "_" + str(num_epochs) + "_"
    model_name_save = model_ID + dataset_ID + '.pt'


    # training
    train_losses = []
    val_losses = []
    loss = 1000
    counter = 0
    first = True
    print("\nstarting training \n")

    # TRAINING LOOP
    for epoch in range(deep_evaluator.n_epochs):

        epoch_train_losses = []

        for x_batch, y_batch in tqdm(train_data):

            counter +=1 # count amount of training batches

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = deep_evaluator.train(epoch, x_batch, y_batch)
            epoch_train_losses.append(loss)
            #train_losses.append(math.log10(loss)) # log loss

            # save first losses before at start of first epoch
            if first:
                first = False
                #train_losses.append(loss)
                #perf_eval = PerfEvaluator(deep_evaluator)
                #precision, val_loss = perf_eval.evaluate(valid_data, verbose = False)
                #val_losses.append(val_loss)

            # for OVERFITTNG 1 element
            #break
            #

        if epoch % 1 == 0:

            train_losses.append(np.mean(epoch_train_losses))

            precision = 0
            val_loss = 0
            # to have the classification reports and confusion_matrix if wanted
            if binary_class:

                perf_eval = PerfEvaluator(deep_evaluator)

                precision, val_loss = perf_eval.evaluate(valid_data, verbose = True)
            else:
                precision, val_loss = perf_eval.evaluate(valid_data, verbose = False)

            #compute val loss
            val_losses.append(val_loss)

            # save the model if the loss on validation set is lower than before
            # on average
            deep_evaluator.saveCheckpoint(val_loss, model_ID + dataset_ID)

            print('Epoch : ', epoch+1, '\t', 'train loss :', train_losses[-1], '\t', 'validation loss : ', val_loss, '\t',  'validation precision : ', precision*100,  " %", "\n")

    x  = list(range(0, len(train_losses)))

    # PLOTS OF LOSSES
    plt.figure()
    plt.plot(x,train_losses, color = "red", label = "training loss")
    plt.plot(x,val_losses, color = "blue", label = "validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Evolution of the validation and training loss ')
    plt.legend()
    plt.savefig("Losses/" + model_ID + dataset_ID + ".png")
    plt.close()

    # SAVING THE MODEL
    torch.save(deep_evaluator.model, 'Models/' + model_name_save )

    # if binary classes, we return the infos for roc curves
    if binary_class:
        fpr, tpr = perf_eval.roc(valid_data)
        return fpr, tpr

    val_precision, val_loss = perf_eval.evaluate(valid_data, verbose = True)
    return val_precision
