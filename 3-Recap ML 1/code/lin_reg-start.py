"""
Punto de partida
"""

import numpy as np
import matplotlib.pyplot as plt

LR = 0.00005
EPOCHS = 20



dataset = np.loadtxt('data/train.csv', delimiter=";")
X = dataset[:,0,None] # Preservar la dimensión y que quede como un vector
y = dataset[:,1,None]

# Estandarizar datos


# Agregar columna de 1 a X para multiplicar por theta_0
m = X.shape[0]
X_0 = np.ones((m,1))
X = np.hstack((X_0,X))

# Parámetros a optimizar
theta = np.random.rand(2)