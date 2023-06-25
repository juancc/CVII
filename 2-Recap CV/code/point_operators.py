"""
Operaciones básicas con imágenes

JCA
"""
import numpy as np
import cv2

def show_im(im, window='window'):
    cv2.imshow(window, im)
    cv2.waitKey(0)

def brightness_contrast(im, alpha, beta):
    return im*alpha + beta

def blend(im1, im2, alpha):
    return (1-alpha)*im1 + alpha*im2

def gamma(im, g):
    return np.power(im, g)



# Brightness
# im = cv2.imread('im/test_im.jpg').astype(float)/255
# im = brightness_contrast(im, 1.5, 0.1)

# Blend
# im1 = cv2.imread('im/day.jpg').astype(float)/256
# im2 = cv2.imread('im/night.jpg').astype(float)/256
# im = blend(im1, im2, 0.75)

# Gamma
# im = cv2.imread('im/test_im.jpg').astype(float)/255
# im = gamma(im, 2)

alpha = cv2.imread('im/test_im-A.jpg').astype(float)/255
im1 = cv2.imread('im/test_im.jpg').astype(float)/255
im2 = cv2.imread('im/sky.jpg').astype(float)/255

im = blend(im2, im1, alpha)


show_im(im)

