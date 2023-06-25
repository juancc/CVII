"""
Encontrar a Waldo usando Correlaciones

JCA
"""

import cv2 as cv
import numpy as np
  
img = cv.imread('data/test_im.png')
kernel = cv.imread('data/waldo.png')
h, w = kernel.shape[:-1]

res = cv.matchTemplate(img, kernel, cv.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img,top_left, bottom_right, 255, 2)

  
cv.imshow('Window', img)
  
cv.waitKey()
cv.destroyAllWindows()