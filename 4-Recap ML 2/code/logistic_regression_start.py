"""
Regresión Logística punto de partida

"""

import numpy as np
import matplotlib.pyplot as plt


LR = 0.02
EPOCHS = 6
BATCH_SIZE = 1000
LAMBDA = 1

def accuracy(h, y):
    m = y.shape[0]
    h[h>=0.5] = 1
    h[h<0.5] = 0
    c = np.zeros(y.shape)
    c[y==h] = 1
    return c.sum()/m


def shuffle(x,y):
    r_indexes = np.arange(len(x))
    np.random.shuffle(r_indexes)

    x = x[r_indexes]
    y = y[r_indexes]

def main():
    # Cargar dataset
    x = np.load('data/x.npy')/255
    y = np.load('data/y.npy')


if __name__ == '__main__': main()