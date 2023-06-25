"""
Ejemplo de Regresión lineal 

JCA
"""
import numpy as np
import matplotlib.pyplot as plt

LR = 0.00005
EPOCHS = 20



dataset = np.loadtxt('data/train.csv', delimiter=";")
X = dataset[:,0,None] # Preservar la dimensión y que quede como un vector
y = dataset[:,1,None]

# Preprocess data - Feature scaling
X = (X- X.mean()) / X.std()
y = (y- y.mean()) / y.std()


# Agregar columna de 1 a X para multiplicar por theta_0
m = X.shape[0]
X_0 = np.ones((m,1))
X = np.hstack((X_0,X))


theta = np.random.rand(2)

def predict(x, theta): 
    return x.dot(theta)


# Test predictions
pred = predict(X[0,:], theta)
print(f'Prediction: {pred}')

def MSE(y, pred): 
    m = y.shape[0]
    return np.power(np.sum(y-pred),2) / (2*m)

# Test MSE
err = MSE(y[0], pred)
print(f'Cost of prediction: {err}')

# Calcular gradientes 
def calc_grad(y, x, theta):
    pred = predict(x, theta)
    x = x[:,1] # Quitar primera columna
    err = y-pred
    g0 = err.sum() / m
    g1 = x.dot(err).sum() / m
    return [g0,g1]

g0,g1 = calc_grad(y, X, theta)
print(f'Gradients: g0:{g0}, g1:{g1}')


def update(theta, grads):
    t0 = theta[0] - LR*grads[0]
    t1 = theta[1] - LR*grads[1]
    return np.array([t0,t1])

theta = update(theta, [g0,g1])
print(f'Update thetas: {theta}')

hist = []

def train(theta):
    for e in range(EPOCHS):
        g0,g1 = calc_grad(y, X, theta)
        theta = update(theta, [g0,g1])
        hist.append(theta)
        err = MSE(y, predict(X, theta))
        print(f'Epoch: {e}. MSE: {err}')

print('Training...')
train(theta)

def plot_training(hist):
    # Plot dataset
    fig, ax = plt.subplots()
    ax.scatter(X[:,1], y, s=0.8)
    i=0
    for t in hist:
        pred = predict(X, t)
        plt.plot(X[:,1], pred, color=[i/EPOCHS, 1,0])
        i +=1
    
    plt.show()

plot_training(hist)