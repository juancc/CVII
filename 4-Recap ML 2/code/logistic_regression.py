"""
Regresión Logística
Clasificación de digitos 0 y 1 de MINST

JCA
"""
import numpy as np
import matplotlib.pyplot as plt


LR = 0.02
EPOCHS = 6
BATCH_SIZE = 1000
LAMBDA = 1

def sigmoid(z):
    """Función de activación Sigmoid"""
    return 1 / (1 + np.exp(-z))

def cost(h,y, theta):
    """Función de costo"""
    m = y.shape[0]
    return -1/m * np.sum( y*np.log(h) + (1-y)*np.log(1-h) ) + LAMBDA/2 * np.power(theta,2).sum()

def feedforward(x, theta):
    z = x.dot(theta)
    return sigmoid(z)


def backpropagation(theta, h, y, x):
    m = y.shape[0]
    d_theta = 1/m * (h-y).dot(x) + LAMBDA/m * theta
    return theta - LR*d_theta 

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
    x = np.load('data/x.npy')/256
    y = np.load('data/y.npy')

    shuffle(x,y)

    # Mostrar una imagen del dataset
    # plt.imshow(x[0], cmap='gray')
    # plt.show()

    # Número de pixeles
    n = x[0].shape[0] * x[0].shape[1] 

    # Número de observaciones
    m = y.shape[0]
    print(f'Dataset con {m} datos')

    # Formatear datos como matriz 2D
    X = x.reshape(m, n)

    # Agregar columna de unos para termino Bias
    x_0 = np.ones((m,1))
    X = np.hstack((x_0,X))

    # Crear un vector para los pesos 
    theta = np.random.rand(n+1) / 10000

    # Probar feedforward
    batch = 5
    _x = X[0:batch] # Vector con valores de 0-1
    _h = feedforward(_x, theta)
    print(_h)

    # Probar función de costo
    _y = y[0:batch]
    print(cost(_h, _y, theta))

    # Entrenamiento
    # Loop por el dataset en batches
    init = 0

    hist = []

    for e in range(EPOCHS):

        for i in range(BATCH_SIZE,m, BATCH_SIZE):
            _x = X[init:i]
            _y = y[init:i]

            h = feedforward(_x, theta)
            j = cost(h, _y, theta)
            
            theta = backpropagation(theta, h, _y, _x)

        # Calcular acc en el dataset de entrenamiento
        acc = accuracy(feedforward(X, theta), y)
        print(f'Epoch: {e}. Cost: {j}. Acc: {acc}')
        hist.append(j)
    
        shuffle(x,y)


    # Graficar entrenamiento
    fig, ax = plt.subplots()
    plt.plot(hist)
    plt.show()






if __name__ == '__main__': main()