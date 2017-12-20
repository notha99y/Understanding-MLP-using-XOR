'''
Helper function to aid the understanding of XOR problem
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

def generate_XOR_distribution(max_x):
    '''
    function take in a max_x (int) and generate a XOR grid.
    returns X ndarray in the input space and Y, the class
    '''
    Sample = 10
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X1_temp = np.linspace(0,max_x,Sample)
    for i in X1_temp:
        for j in X1_temp:
            X1.append([i,j])
            X2.append([-i,-j])
            X3.append([-i,j])
            X4.append([i,-j])
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)

    X = np.vstack([X1,X2,X3,X4])
    Y = np.array([0] * 2*Sample **2 + [1] *2* Sample**2)
    return X, Y
    
def build_mlp_model(dims):
    '''
    build multi-layer perceptron network according to dims

    dims is an array of integers specifiying input, hidden and output layers
    '''
    model = Sequential()
    model.add(Dense(dims[1], activation = 'sigmoid', input_dim = dims[0]))
    for dim in range(1,len(dims) - 2):
        model.add(Dense(dims[dim+1], activation = 'sigmoid'))
        # model.add(Dropout(rate = 0.5))
    model.add(Dense(dims[-1], activation = 'sigmoid'))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])
    return model

def accuracy_loss_plot(history):
    '''
    Accuracy and loss plot for mlp trained in keras

    history is model.fit

    shows the graphs
    '''
    # list all data in history
    # print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
#     plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
#     # summarize history for loss
    plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
#     plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
# train_x, train_y = preprocessing_model_1()
# print(train_x.shape)
# print(train_y.shape)
