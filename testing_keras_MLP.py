import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from data_utils import generate_XOR_distribution, accuracy_loss_plot

# Generating XOR
X,Y = generate_XOR_distribution(1)

# Indicating the nodes
n_input = 2
n_hidden = 10
n_output = 2

#Instantiating the model
model = Sequential()

#Adding the layers
model.add(Dense(n_hidden, activation='sigmoid', input_dim=n_input))
model.add(Dense(n_output, activation='sigmoid'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=0.01),
             metrics = ["accuracy"])

print("MLP model summary: ",model.summary())

# One hot encoding
Y_out = keras.utils.to_categorical(Y,num_classes=2)

# Training
history = model.fit(X,Y_out,epochs = 1000,batch_size = 400)

# Plotting
accuracy_loss_plot(history)
