"""
Ejercicio de transformaciones 2D

JCA
"""
import numpy as np
import cv2 as cv

def draw_triangle(im, triangle, color=(0,255,0)):
    # Normalizar coordenadas homogeneas
    triangle = np.array([ v[:-1]/v[-1] for v in triangle] , np.uint16)
    cv.drawContours(im, [triangle.astype(int)], 0, color, -1)
    for v in triangle:
        cv.circle(im, tuple(v), 2, (255,0,255),-1)

def scale(vertex, sx, sy):
    S = np.array([[sx,0,0], [0,sy,0], [0,0,1]])
    return np.matmul(S, vertex.T).T

def rotate(vertex, a):
    a *= np.pi/180
    R = np.array([[np.cos(a),-np.sin(a),0], [np.sin(a),np.cos(a),0], [0,0,1]])
    return np.matmul(R, vertex.T).T

def translate(vertex, dx, dy):
    T = np.array([[1,0,dx], [0,1,dy], [0,0,1]])
    res =  np.matmul(T, vertex.T) # Transponer vertices en columnas
    return res.T

def show_im(im, window='window'):
    cv.imshow(window, im)
    cv.waitKey(0)

# image size
w = 500
h = 500
im = np.zeros((h,w,3), np.uint8)

# Parametros de las transformaciones
# Traslada a centro y realiza escala y rotacion
angle = 30
s = 3
t = 250
# Sistema coordenado de imagenes
triangle1 = np.array( [[10,10,1], [70,10,1], [40, 60,1]])
triangle2 = translate(rotate( scale(translate(triangle1, -40,-30),s,s) ,angle), t,t)

# print(triangle2)
draw_triangle(im, triangle1)
draw_triangle(im, triangle2, color= (0,100,255))

show_im(im)