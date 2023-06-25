"""Visión por computador II - Introducción
Cambiarle el fondo a la imagen

JCA
"""
import numpy as np
import cv2


def show_im(im):
    cv2.imshow('window', im)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


im = cv2.imread('/Users/juanca/Library/Mobile Documents/com~apple~CloudDocs/Courses/Computer Vision II/Contenido/Introducción/code/cat.jpg')

# El orden de los canales es: BGR
# green = im[:,:,1]
im[im[:,:,1] > 130] = [100,200,200]

show_im(im)