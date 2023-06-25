"""
Transformaciones geométricas
Punto de partida

JCA
"""
import numpy as np
import cv2 as cv

W = 500
H = 500

im = np.zeros([W,H,3])

triangle =  np.array( [[10,10], [70,10], [40, 60]])

def scale(triangle, sx,sy):
    t = np.array([[sx,0],[0,sy]])
    return triangle @ t

def translate(triangle, dx, dy):
    t = np.array([dx,dy])
    return triangle + t

def draw(triangle, im, color=(0,255,0)):
    cv.drawContours(im, [triangle.astype(int)], 0, color, -1)


def show_im(im):
    cv.imshow('window', im)
    cv.waitKey(0)

draw(triangle, im)
t = translate(triangle, 100, 10)
draw(t, im)

show_im(im)