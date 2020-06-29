# PERCEPTRON IMPLEMENTING a SINGLE-TLU
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X    = iris.data[:,(2,3)]   # petal length, petal width
y    = (iris.target == 0).astype(np.int)    # Iris setosa

per_clf = Perceptron()
per_clf.fit(X,y)

y_pred = per_clf.predict([[2, 0.5]])
y_pred
# Single Precerptron has a XOR classification problem

# MULTILAYER PERCEPTRON (MLP)
# IMPLEMENTING MLPs with Keras
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

# BUILDING AN IMAGE CLASSIFIER USING the SEQUENTIAL API
# Load the Fashion MNIST Dataset with Keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
print(X_train_full.dtype)

# Validation Set & Feature Scaling (Diveided by 255 to scale the pixel indentities)
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Class Name of y
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
class_names[y_train[0]]

# Create Model with the Sequential API
# Classification MLPs with 2 Hidden Layers
model = keras.models.Sequential()
#model.add(keras.layers.InputLayer(input_shape = [28,28]))
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))
# OR
#model = keras.models.Sequential([
#        keras.layers.Flatten(input_shape = [28,28]),
#        keras.layers.Dense(300, activation = "relu"),
#        keras.layers.Dense(100, activation = "relu"),
#        keras.layers.Dense(10, activation = "softmax")
#        ])
model.summary()
model.layers
hidden1 = model.layers[1]
hidden2 = model.layers[2]
hidden1.name
hidden2.name
model.get_layer('dense') is hidden1

weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)
print(biases)
print(biases.shape)

# COMPILE MODELS
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])

# TRAINING & EVALUATING MODELS
history = model.fit(X_train, y_train, epochs = 30,
                    validation_data = (X_valid, y_valid))

# History 
print(history.params)
print(history.epoch)
print(history.history)
# LEARNING CURVES
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca.set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# (Assume) AFTER TUNING, TRY ON TEST DATASET
model.evaluate(X_test, y_test)

# MAKE PREDICTIONS
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
y_pred
np.array(class_names)[y_pred]

y_new = y_test[:3]
y_new
np.array(class_names)[y_new]


# BUILDING a REGRESSION MLP with SEQUENTIAL API
# Load the California Dataset with sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import os

#HOUSING_PATH  = os.path.join("datasets","housing")
#csv_path = os.path.join(HOUSING_PATH,"housing.csv")
#housing = pd.read_csv(csv_path)
housing  = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,
                                                              housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, 
                                                      y_train_full)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test  = scaler.fit_transform(X_test)

model = keras.models.Sequential([
        keras.layers.Dense(30, activation = "relu", input_shape = X_train.shape[1:]),
        keras.layers.Dense(1)
        ])
model.compile(loss = "mean_squared_error", optimizer = "sgd")
history = model.fit(X_train, y_train, epochs = 20, 
                    validation_data = (X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)


# BUIDLING COMPLEX MODELS with FUNCTIONAL API

# Wide & Deep Model on Cali. Housing Dataset
input_  = keras.layers.Input(shape = X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation = "relu")(input_)
hidden2 = keras.layers.Dense(30, activation = "relu")(hidden1)
concat  = keras.layers.Concatenate()([input_, hidden2])
#concat = keras.layers.concatenate([input_, hidden2])
output  = keras.layers.Dense(1)(concat)

model = keras.Model(inputs = [input_], outputs = [output])

model.compile(loss = "mean_squared_error", optimizer = "gd")
history = model.fit(X_train, y_train, epochs = 20, 
                    validation_data = (X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
# Problem: get loss:nan when fitting 
'''Output Sample
Train on 11610 samples, validate on 3870 samples
Epoch 1/20
11610/11610 [==============================] - 1s 86us/sample - loss: 0.7514 - val_loss: 0.5255
Epoch 2/20
11610/11610 [==============================] - 0s 39us/sample - loss: 1.1747 - val_loss: 0.9530
Epoch 3/20
11610/11610 [==============================] - 0s 37us/sample - loss: 1.9128 - val_loss: 1.0357
Epoch 4/20
11610/11610 [==============================] - 0s 38us/sample - loss: 2.8036 - val_loss: 1.2471
Epoch 5/20
11610/11610 [==============================] - 0s 38us/sample - loss: 0.6055 - val_loss: 1.3445
Epoch 6/20
11610/11610 [==============================] - 0s 37us/sample - loss: 3.4298 - val_loss: 1.5355
Epoch 7/20
11610/11610 [==============================] - 0s 39us/sample - loss: 0.6756 - val_loss: 1.4621
Epoch 8/20
11610/11610 [==============================] - 0s 39us/sample - loss: nan - val_loss: nan
Epoch 9/20
11610/11610 [==============================] - 0s 38us/sample - loss: nan - val_loss: nan
Epoch 10/20
11610/11610 [==============================] - 0s 37us/sample - loss: nan - val_loss: nan
Epoch 11/20
11610/11610 [==============================] - 0s 38us/sample - loss: nan - val_loss: nan
Epoch 12/20
11610/11610 [==============================] - 0s 39us/sample - loss: nan - val_loss: nan
Epoch 13/20
11610/11610 [==============================] - 0s 37us/sample - loss: nan - val_loss: nan
Epoch 14/20
11610/11610 [==============================] - 0s 37us/sample - loss: nan - val_loss: nan
Epoch 15/20
11610/11610 [==============================] - 0s 38us/sample - loss: nan - val_loss: nan
Epoch 16/20
11610/11610 [==============================] - 0s 38us/sample - loss: nan - val_loss: nan
Epoch 17/20
11610/11610 [==============================] - 0s 37us/sample - loss: nan - val_loss: nan
Epoch 18/20
11610/11610 [==============================] - 0s 38us/sample - loss: nan - val_loss: nan
Epoch 19/20
11610/11610 [==============================] - 0s 38us/sample - loss: nan - val_loss: nan
Epoch 20/20
11610/11610 [==============================] - 0s 39us/sample - loss: nan - val_loss: nan
5160/5160 [==============================] - 0s 19us/sample - loss: nan
'''
# Seems that it doesn't converge, try to adjust optimizer

# MULTIPLE INPUTS MODEL
# 5 features through the wide 0 to 4, 6 features 2 to 7 through the deep
input_A = keras.layers.Input(shape = [5], name = "wide_input")
input_B = keras.layers.Input(shape = [6], name = "deep_input")
hidden1 = keras.layers.Dense(30, activation = 'relu')(input_B)
hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)
concat  = keras.layers.concatenate([input_A, hidden2])
output  = keras.layers.Dense(1, name = "output")(concat)

model = keras.Model(inputs = [input_A, input_B], outputs = [output])
model.compile(loss = "mse", optimizer = keras.optimizers.SGD(lr = 1e-3))

X_train_A, X_train_B = X_train[:,:5], X_train[:,2:]
X_valid_A, X_valid_B = X_valid[:,:5], X_valid[:,2:]
X_test_A, X_test_B = X_test[:,:5], X_test[:,2:]
X_new_A, X_new_B = X_test_A[:5], X_test_B[:5]

history = model.fit((X_train_A, X_train_B), y_train, epochs = 20,
                    validation_data = ((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

