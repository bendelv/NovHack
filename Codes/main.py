# this script uses the NetLauncher.py function to launch networks. It uses some Hyperparameters
# that are defined at the top of the script, such as the path to the various parts of the dataset used,
# the name of the model ( referencing one of the models architecture that must be in the Nets.py file)
# the numbr of epochs, if the network will end up predicting binary classes or not,
# the mini batches size, the dropout rate, etc...
# this results in a model being trained on the referenced training set and saved in the
# Models folder. At the end of each epoch, if the current version of the model is performing better on the
# validation set than the previous best model version, then the current version will be saved as a checkpoint.
# furthermore, the plot of the evolution of the validation and training loss will be saved in the Loss folder.

# finally, the Hyperparameters can be studied through a grid search by referencing several values in their corresponding lists.

from NetLauncher import netLauncher
import matplotlib.pyplot as plt

# data paths normalized 35 sec  with 5sec of sliding and new valid windows rules
train_path = 'data_batch_1'
valid_path = 'data_batch_2'
test_path  = 'test_batch'

# Hyperparameters
model_name = "CNN9"
num_epochs = 2
binary_class = False

#hyperparam of grid search
batches = [64] #?
dropouts = [0] # only for regularization ?
weights = [False]
lr = [0.0001] # ?

# best model eval
best_val_precision = 0
best_batch = 0
best_dropout =  0
best_weight = 0
best_lr = 0


ID = ""
spl = train_path.split('/')
dataset_ID = spl[0]


# grid search
for batch_size in batches:
    for dropout_rate in dropouts:
        for weighted in weights:
            for learning_rate in lr:
                print("\n NEW MODEL \n")
                print(" batch_size = ", batch_size, "\n dropout = ", dropout_rate,
                      "\n weights = ", weighted, "\n learning_rate = ", learning_rate, " \n \n")


                # if binary classes, ROC curves
                if binary_class:
                    fpr, tpr = netLauncher(num_epochs, batch_size, learning_rate, dropout_rate, weighted, model_name, train_path, valid_path,
                                    test_path, binary_class)

                    # unique string identifying the model
                    ID = model_name + "_" + str(batch_size) + "_" + str(dropout_rate) + "_" + str(weighted) + "_" + str(learning_rate)
                    # update ROC plot
                    # plt.plot(fpr, tpr, label= ID + ".png")

                else:
                    val_precision = netLauncher(num_epochs, batch_size, learning_rate, dropout_rate, weighted, model_name, train_path, valid_path,
                                    test_path, binary_class)

                    if val_precision > best_val_precision:
                        best_val_precision = val_precision
                        best_batch =  batch_size
                        best_dropout =  dropout_rate
                        best_weight =  weighted
                        best_lr =  learning_rate
