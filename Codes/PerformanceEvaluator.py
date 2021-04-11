from DeepEvaluator import DeepEvaluator
from torch.autograd import Variable
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, plot_precision_recall_curve, auc
from sklearn.metrics import roc_curve
import numpy as np


# class containing tools to evaluate a pytorch classifier model
class PerfEvaluator():
    def __init__(self, model):
        self.model = model
        self.classes = model.classes

    def displayClassDiversity(self, set):
        i = 0
        data_div = None
        if set == "training":
            data_div = self.model.training_data_diversity
            print("\n TRAINING SET DATA DISTRIBUTION\n")
        if set == "testing":
            data_div = self.model.testing_data_diversity
            print("\n TESTING SET DATA DISTRIBUTION\n")
        if set == "validation":
            data_div = self.model.validation_data_diversity
            print("\n VALIDATION SET DATA DISTRIBUTION\n")

        for cl in data_div:
            string = self.classes[i] + ': ' + str(cl)
            print(string)
            i += 1

    # evaluate the model on the data provided. data is assumed to come from DataLoader.
    # returns precision and print full report if verbose = True
    def evaluate(self,data, verbose):

        if verbose:
            self.displayClassDiversity("training")

        correct = 0
        total = 0
        classes =  [0] * len(self.classes)
        batches_losses = []

        # Initialize the prediction and label lists(tensors)
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        for x_batch, y_batch in data:

        # FOR OVERFIT TESTS
        #for x_batch, y_batch in data:

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                x_batch = Variable(x_batch).unsqueeze(1).float()
                y_batch = Variable(y_batch)

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = self.model.model(x_batch)
                #outputs = self.model.best_net(x_batch) : changed, maybe error

                # computing valid loss
                batch_loss = self.model.criterion(outputs, y_batch)
                batches_losses.append(batch_loss.item())

                # indices of the predicted classe
                _, predicted = torch.max(outputs.data, 1)

                # Append batch prediction results
                predlist=torch.cat([predlist,predicted.view(-1).cpu()])
                lbllist=torch.cat([lbllist,y_batch.view(-1).cpu()])

                # conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
                # precision, recall, thresholds = precision_recall_curve(lbllist.numpy(), predlist.numpy())

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()


                # FOR OVERFIT TESTS
                #break

        if verbose:

            # self.displayClassDiversity("validation")
            # print("\n VALIDATION  RESULTS")
            # print("total validation data= " + str(total))
            # print("correct predictions = " + str(correct))
            # print("precision = " + str(correct/total))
            #
            # #Confusion matrix
            # print("\nConfusion Matrix\n")
            # string = " "
            # for cl in self.classes:
            #     string += " " + cl
            # print(string)
            # print(conf_mat)
            #
            # # Per-class accuracy
            # print("\nPER CLASS ACCURACY")
            # data_div = self.model.validation_data_diversity
            # class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
            #
            # for i in range(len(self.classes)):
            #     string = self.classes[i] + ': ' + str(class_accuracy[i])
            #     print(string)

            conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
            print(conf_mat)
            TP = conf_mat[1][1]
            FN = conf_mat[1][0]
            FP = conf_mat[0][1]
            print("recall : " + str( TP/(TP+FN) ) )
            print("precision : " + str(TP/(TP+FP)))


        return (correct/total), np.mean(batches_losses)

    def roc(self, valid_data):

        # Initialize the prediction and label lists(tensors)
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        for patient_data in valid_data:
            # evaluate one test data
            for x_batch, y_batch in patient_data:

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                x_batch = Variable(x_batch).unsqueeze(1).float()
                y_batch = Variable(y_batch).long().squeeze(1)

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = self.model.best_net(x_batch)
                # indices of the predicted classe
                _, predicted = torch.max(outputs.data, 1)

                # Append batch prediction results
                predlist=torch.cat([predlist,predicted.view(-1).cpu()])
                lbllist=torch.cat([lbllist,y_batch.view(-1).cpu()])

        fpr, tpr, _ = roc_curve(lbllist, predlist)
        return fpr, tpr
