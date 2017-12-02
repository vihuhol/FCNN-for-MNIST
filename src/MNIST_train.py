import cv2
import download_mnist
import numpy as np
from NeuralNetwork import myMLPClassifier
from matplotlib import pyplot as plt

N_train = 60000
N_test = 10000

print('Load data...')
train_ims = download_mnist.train_images()[:N_train]
train_labs = download_mnist.train_labels()[:N_train]
test_ims = download_mnist.test_images()[:N_test]
test_labs = download_mnist.test_labels()[:N_test]

print('Preprocessing...')
X_train = np.zeros((N_train, 28*28))
for I,image in enumerate(train_ims):
    X_train[I] = image.flatten()
X_test = np.zeros((N_test, 28*28))
for I,image in enumerate(test_ims):
    X_test[I] = image.flatten()
y_train = train_labs;
y_test = test_labs

print('Training...')
clf = myMLPClassifier(hidden_layer_sizes=(800, ), random_state = 2, task='classification', tol=0.001, max_iter=100, batch_size = 10)
clf.fit(X_train, y_train)

print('Predicting...')
y_train_predict = clf.predict(X_train)
y_test_predict = clf.predict(X_test)

print('Errors calc...')
err_train = 0
for i in range(N_train):
    if y_train_predict[i] != y_train[i]:
        err_train += 1.0
err_train = err_train / N_train
print(err_train)
err_test = 0
for i in range(N_test):
    if y_test_predict[i] != y_test[i]:
        err_test += 1.0
err_test = err_test / N_test
print(err_test)


