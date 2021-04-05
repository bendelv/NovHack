import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Linear, Sequential, ReLU, Conv1d, MaxPool1d, BatchNorm1d, Module, CrossEntropyLoss, MSELoss
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

from DataExtractor import DataExtractor
from Nets import FC4,CNN9

import matplotlib.pyplot as plt

import numpy as np

from time import sleep

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)


# this class represents a NN model and encompasses a variety of tools that can be used to train the model in various ways.
# PARAMETERS :
# nb_epochs :  number of epochs the model will train on
# learning_rate :  rate used to update weights in backpropagation
# dropout_rate : dropout rate
# weighted : boolean indicating wether or not the various classes the model tries to classify are being weighted when computing  the loss
# model_name : name of the specific model architecture one wants to build. The corresponding achitecture must be implemented in the Net.py file
# binary class : boolean indicating wether the model predicts binary classes or not

class DeepEvaluator():
    def __init__(self, nb_epochs, learning_rate, dropout_rate, weighted, model_name, binary_class):
        self.model = self.modelFromName(model_name, dropout_rate)

        # net architecture
        print(self.model)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = CrossEntropyLoss()
        self._best_loss =  1
        self.best_net = self.modelFromName(model_name, dropout_rate)
        # boolean. If true, weights are applied to the classes. If not, weights = 1
        self.weighted = weighted

        # defining the number of epochs
        self.n_epochs = nb_epochs

        self.classes = ('sleep', 'Rva',
           'Hpop', 'CA', 'Aobs', 'Limd', 'Amix', 'Micev', '9', '10')

        if binary_class:
            self.classes = ('sleep', 'micev')

        self.training_data_diversity = [0] * len(self.classes)
        self.validation_data_diversity = [0] * len(self.classes)
        self.testing_data_diversity = [0] * len(self.classes)


    # this function takes the name and dropout_rate of a model and return the
    # actual implementation of this model.

    # ARGUMENTs :
    # model_name : string representing the name of a model architecture that will be imported from Nets.py
    # dropout_rate : dropout rate used by the model
    # RETURNS : a pytorch model
    def modelFromName(self, model_name, dropout_rate):
        if model_name == "CNN_Net":
            return CNN_Net(dropout_rate).to(device)

        if model_name == "Julien_CNN":
            return Julien_CNN(dropout_rate).to(device)

        if model_name == "PaperCNN":
            return PaperCNN(dropout_rate).to(device)

        if model_name == "CNN9":
            return CNN9(dropout_rate).to(device)

        if model_name == "FC4":
            return FC4(dropout_rate).to(device)



    # counts the amount of samples of each class present in the tensor input.
    # returns a list of length "nb_of_classes" where the ith element of the list
    # cointains the amount of elements of class i
    def dataDiversityFromTensors(self,tensors):
        repartition = [0] * len(self.classes)
        for tensor in tensors:
            for y in tensor:

                class_index = int(y.item())
                repartition[class_index] =  repartition[class_index] + 1

        return repartition

    # Makes checkpoints of the current model if it is the best performance yet.
    # compare the current loss to the best loss met so far and if the current loss is better,
    # saves the current state of the model as the best model.

    # ARGUMENTS :
    # loss :  current loss of the current model (expected to be better if closer to 0)
    # model_name :  full and unique name identifying the model and the conditions it was trained with.
    # Will be used as the name of the saved file.
    def saveCheckpoint(self, loss, model_name):
        """Save checkpoint if a new best is achieved"""
        # update current best loss
        if loss < self._best_loss:
            print("new best model ! ")
            torch.save(self.model, 'Models/checkpoint_' + model_name + '.pt')
            self.best_net = self.model
            self._best_loss = loss

    # from an X and Y .mat path, generates corresponding X and Y matrices, then feeds them
    # in a TensorDataset and then return a DataLoader. Some informations about the classes found in the sets will be saved for
    # potential class diversity analysis
    # the DataLoader returned will have a WeightedRandomSampler if the set is a training set. In that case,
    # the batches produced by the DataLoader will have some corrected balance between the majority class and the minority one

    # ARGUMENTS :
    # X_path is the path to the  .mat file that contains the X values
    # Y_path is the path to the  .mat file that contains the Y values
    # batch_size is the size each mini batch in the DataLoader should have
    # set is a string that can take three values : "training", "validation" and "testing". It precises the usage of the data being loaded
    # binary_class :  boolean indicating wether or not the dataset should consider just 2 ckasses or the total diversity

    # RETURNS :
    # a pytorch DataLoader object
    def loadDataset(self, X_path, Y_path, batch_size, set, binary_class):

        # loading data !!! HACKATHON : extraire les data de facon appropriée ici
        X,Y = DataExtractor(X_path, Y_path)

        # need to format data from numpy ndarray into tensors for TensorDataset
        X = torch.tensor(X)
        y = torch.tensor(Y)

        if set == "training":

            self.training_data_diversity = self.dataDiversityFromTensors([y])

            # adapting class weights as multiples of the more numerous class
            max_index = np.argmax(self.training_data_diversity)
            proportion_max_class = self.training_data_diversity[max_index]/np.sum(self.training_data_diversity)
            print("MAJORITY CLASS :  " + self.classes[max_index] + ", " + str(proportion_max_class*100) + "%")
            biggest_class_size = self.training_data_diversity[max_index]
            weights =  [1] * len(self.classes)

            # if the weighted options has been chosen, all classes get weighted with a ratio that is the inverse of their proportion
            # in the dataset
            if self.weighted :
                for i in range(len(self.training_data_diversity)):

                    weights[i] =  biggest_class_size/self.training_data_diversity[i]

            print("CLASS WEIGHTS ")
            print(weights)

            class_weights = torch.FloatTensor(weights).to(device)
            self.criterion = CrossEntropyLoss(weight=class_weights)

        if set == "testing":
            # testing sets are build from several individual files : need to count the diversity of classes one file
            # after another
            new_list = self.dataDiversityFromTensors([y])
            previous_list = self.testing_data_diversity
            res_list = [previous_list[i] + new_list[i] for i in range(len(new_list))]
            self.testing_data_diversity = res_list


        if set == "validation":
            # validation sets are build from several individual files : need to count the diversity of classes one file
            # after another
            new_list = self.dataDiversityFromTensors([y])
            previous_list = self.validation_data_diversity
            res_list = [previous_list[i] + new_list[i] for i in range(len(new_list))]
            self.validation_data_diversity = res_list


        X.to(device)
        y.to(device)

        # TensorDataset is used to feed the data under tensor form to DataLoader
        data = TensorDataset(X,y)

        # valid and test sets should not have modified proportions of classes in the batches
        if set !=  "training":
            # sets the data in batches, takes advantage of parallel processing i think ?
            data = DataLoader(
                dataset=data, batch_size=batch_size, shuffle=False)

            return data

        # balanced sampling in the batches (controlled proportion of the samples of class 1 and 0 in each batch)
        labels = np.array(Y[:,0])# all individual samples y values
        labels = labels.astype(int) # get labels of data in int format

        # HACKATON here need to reread doc and attribute weights that would make sense in our case
        weight_majority = 2 / self.training_data_diversity[0] # weight attributed to samples from the class of the majority
        weight_minority = 1/ self.training_data_diversity[1] # weight attributed to samples from the class of the minority

        samples_weights = np.array([weight_majority, weight_minority]) # each individual sample of the data is attributed a weight
        weights = samples_weights[labels] # each individual sample gets the weight attributed to its label

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples = len(labels), replacement = True)

        # sets the data in batches, takes advantage of parallel processing i think ?
        data = DataLoader(
            dataset=data, batch_size=batch_size, shuffle=False, sampler = sampler)

        return data

    # train function of the pytorch model
    def train(self, epoch, train_X, train_y):
        self.model.train()

        # getting the training set
        X_train = Variable(train_X).unsqueeze(1).float()
        y_train = Variable(train_y).long().squeeze(1)

        # prediction for training and validation set
        output_train = self.model(X_train)

        # computing the training and validation loss
        # the output has to be in the format [batch_size, nb_of_classes]
        # the target has to be in the format [batch_size]
        loss_train = self.criterion(output_train, y_train)

        # empty the gradients
        self.optimizer.zero_grad()

        # computing the updated weights of all the model parameters
        loss_train.backward()

        self.optimizer.step()

        return loss_train.item()
