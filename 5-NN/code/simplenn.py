# -*- coding: utf-8 -*-
"""simpleNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-5TLXVdSIFOKYhDkwDTTYsanH4kkhglm

# Red Neuronal estilo funcional
Implementación básica de una red neuronal usando python
"""

"""
MNIST database of handwritten digits
Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.


Returns:

    2 tuples:
        x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
        y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).


"""
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255

"""
Mostrar algunos datos del dataset
"""

from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
for i in range(1,26):
    plt.subplot(5,5,i)
    plt.title(y_train[i])
    plt.subplots_adjust(top=1.5)
    plt.axis('off')
    plt.imshow(x_train[i], 'gray')

"""# Arquitectura de la Red
*   Entrada 28x28
*   Oculta 25
*   Salida 20

![alt text](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fdocs.opencv.org%2F2.4%2F_images%2Fmlp.png&f=1&nofb=1)



"""

# Arquitectura de la red
input_size = x_train.shape[1] *x_train.shape[2]#28x28
hidden_size = 25
output_size = 10

"""Función de activación"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

_x = np.czz(-10,10,1)
_y = np.array([sigmoid(i) for i in _x])
plt.plot(_x,_y)

import numpy as np
def log(n):
    return np.where(n<0, 12, np.log(n))

def cost_func(h,Y):
    m = Y.shape[0]
    return  1/m * np.sum( np.sum( -Y * np.log(h) - (1-Y) * np.log(1 - h) )) 



h = np.array([[0.99999,0.000001]])
y = np.array([[0.99999,0.000001]])

cost_func(h,y)

def feedforward(X, w1, w2):
    a1 = sigmoid(np.dot(X, w1))
    h = sigmoid(np.dot(a1, w2))
    return a1, h

def rand_weights(l_in, l_out):
    epsilon_init = 0.12;
    W = np.random.rand(l_in, l_out) * 2 * epsilon_init - epsilon_init;
    return W

# Valores iniciales
weights1   = rand_weights(input_size, hidden_size) 
weights2   = rand_weights(hidden_size, output_size) 

print(weights1.shape)
print(weights2.shape)

# Probar feedforward
m = 5
# _sample = x_train[0:5].reshape(1, input_size)/255 # Vector con valores de 0-1
_sample = x_train[0:m].reshape(m, input_size)
print(_sample.shape)
_l1, _res = feedforward(_sample, weights1, weights2)
print(_res)

def _backpropagation(y, output, w2, l1, w1, X):
    z1 = np.dot(X, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    h = sigmoid(z2)

    d_sig_out = d_sigmoid(z2)
    d_err = y - h
    
    d_weights2 = np.dot(a1.T, (d_err * d_sig_out))
    d_weights1 = np.dot(X.T, (np.dot(d_err * d_sig_out, w2.T) * d_sigmoid(a1)))

    w1 += d_weights1
    w2 += d_weights2

    return w1, w2
def backpropagation(Y, output, w2, l1, w1, X):
    m=X.shape[0]
    z1 = np.dot(X, w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2)
    h = sigmoid(z2)

    d3 =  h - Y
    d2 = np.dot(d3, w2.T) * d_sigmoid(z1)

    w2_grad = 1/m * (d3.T * a1)
    w1_grad = 1/m * np.dot(d2.T, X)

    w2 += w2_grad.T
    w1 += w1_grad.T
    
    return w1, w2


# d3 = h-Y; % error of layer 3 - hypothesys 
# d2 = Theta2' * d3' .* sigmoidGradient( [ones(m,1) z2]' );


# t2 = Theta2; t2(:,1) = 0; %creating thetas without bias row for regulation term
# t1 = Theta1; t1(:,1) = 0;

# Theta2_grad = 1/m * ( d3' * [ones(m,1) a2] ) + (lambda/m) * t2; 
# Theta1_grad = 1/m * ( d2(2:end, :) * X) + (lambda/m) * t1;

def encode_label(y):
    _y = np.zeros(10)
    _y[y] = 1 # Codificar respuesta objetivo
    return _y

# Probar backpropagation
m = 1
# _sample = x_train[0:m].reshape(m, input_size)/255 # Vector con valores de 0-1
_sample = x_train[0].reshape(m, input_size)

weights1   = rand_weights(input_size, hidden_size) 
weights2   = rand_weights(hidden_size, output_size) 

_l1, _res = feedforward(_sample, weights1, weights2)

_y = encode_label(y_train[0])

w1_init = np.array(weights1, copy=True)
w2_init = np.array(weights2, copy=True)

w1, w2 = backpropagation(_y, _res, weights2, _l1, weights1, _sample)
print('W1')
print(w1.shape)
print(w1 == w1_init)
print('W2')
print(w2.shape)
print(w2 == w2_init)

# Entrenamiento


def train(x_train, y_train, iterations):
    w1   = rand_weights(input_size, hidden_size) 
    w2   = rand_weights(hidden_size, output_size) 

    np.random.seed(1)

    for i in range(iterations):
        print('It: {}'.format(i))
        # np.random.shuffle(x_train)
        # np.random.shuffle(y_train)
        err = 0

        for t in range(x_train.shape[0]):
            x = x_train[t].reshape(1, input_size)
            l1, res = feedforward(x, w1, w2)

            y = encode_label(y_train[t])

            w1, w2 = backpropagation(y, 0, w2, 0, w1, x)

            res = res.tolist()
            res = res.index(max(res))
            err +=  (res - y_train[t])**2 
        
        print('Err: {}'.format( err/x_train.shape[0]))

    return w1,w2


w1, w2 = train(x_train[0:5], y_train[0:5], 1)

# Predecir con pesos entrenados
idx = 1


x = x_test[idx].reshape(1, input_size)
_, res = feedforward(x, w1, w2)

plt.imshow(x_test[idx], 'gray')

res = res.tolist()
res = res.index(max(res))
print(res)