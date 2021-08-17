# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Model configuration (Hyperparameters)
batch_size = 25
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 40
learning_rate = 0.1
decay_rate = learning_rate / no_epochs
optimizer = SGD()
verbosity = 1
num_folds = 20

# Load CIFAR-10 data
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
for train, test in kfold.split(inputs, targets):

  model = keras.Sequential([
      # cnn 
      layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)),                  
      layers.MaxPooling2D((2,2)),

      # Dense 
      layers.Flatten(),
      layers.Dense(264, activation='relu'),
      layers.Dense(264, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])

  print("----------------------------------------------------------------------------")
  print(f"Training for fold {fold_no} ...")
  sgd = SGD(learning_rate=learning_rate, decay=decay_rate, nesterov=False)

  model.compile(
      optimizer=sgd,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )

  model.fit(inputs[train], targets[train], epochs=no_epochs, verbose=True)

  eval_loss, eval_accuracy = model.evaluate(inputs[test], targets[test], verbose=False)
  print(f"Model loss is: {eval_loss} accuracy is {eval_accuracy}")

  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  fold_no = fold_no + 1

model.save('CIFAR 10 LEO')

model = keras.models.load_model('CIFAR 10 LEO')
predict = model.predict(input_test)
print(f"Prediction shape is {predict.shape}")

for i in range(0, len(acc_per_fold)):
  print(f'Loss for fold {i+1} is {loss_per_fold[i]} and accuracy is {acc_per_fold[i]}')
  print('----------------------------------------------------------------------------')
  
print(f'Accuracy for CV is {sum(acc_per_fold)/len(acc_per_fold)} ')
print(sum(acc_per_fold))
print(len(acc_per_fold))
print(acc_per_fold)
print(f'Loss for CV is {np.mean(loss_per_fold)} ')

eval_loss, eval_accuracy = model.evaluate(input_test, target_test, verbose=False)
print(f"Model accuracy is {eval_accuracy}")
