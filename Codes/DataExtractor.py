
import numpy as np
import h5py
import os.path

# from a X and Y dataset path, create 2 numpy array of X and Y values
def DataExtractor(X_path, Y_path):

    X = h5py.File(X_path, 'r')
    Y = h5py.File(Y_path, 'r')

    for k,v in X.items():
        X_out = np.array(v).T

    for k,v in Y.items():
        Y_out = np.array(v).T

    return [X_out,Y_out]

# from a X dataset, create a numpy array of X values
def XDataExtractor(X_path):

    X = h5py.File(X_path, 'r')

    for k,v in X.items():
        X_out = np.array(v).T

    return [X_out]

# from a valid general valdiX  and  validY paths, generate concatenated X and Y validation set
def BuildValidOrTestSet(validX_path, validY_path):

    # buidling validation set
    X_valid = [[]]
    Y_valid = [[]]
    first_valid = True

    for i in range(95):
        valid_f_X_path = validX_path.format(i)
        valid_f_Y_path = validY_path.format(i)
        if os.path.isfile(valid_f_X_path):

            if first_valid:
                first_valid =False
                X_valid, Y_valid = DataExtractor(valid_f_X_path, valid_f_Y_path)
                continue

            X, Y = DataExtractor(valid_f_X_path, valid_f_Y_path)

            X_valid = np.concatenate((X_valid, X), axis=0)
            Y_valid = np.concatenate((Y_valid, Y), axis=0)

    return X_valid,Y_valid
