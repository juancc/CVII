"""Visión por computador II - Introducción
Manejo de imágenes: dibujar en pixeles

JCA
"""
import numpy as np
import cv2

W = 100
H = 100


def im_1():
    
    im = np.zeros((W,H,3), np.uint8)
    for i in range(0,100): im[i,i] = [255,255,255]
    cv2.imwrite('im_1.png', im)


def draw_square(im, side, color):
    """Función que dibuja cuadros en el centro
        :param im: (np.array) imagen en la que dibuja
        :param side: (int) lado del cuadrado a dibujar
        :para color: (list or tuple) color del cuadrado
    """
    c = W/2
    ini = c - side/2
    end = c + side/2
    for i in range(W):
        for j in range(H):
            if i> ini and j>ini and i<end and j<end:
                im[i,j] = color


def im_2():
    im = np.zeros((W,H,3), np.uint8)
    draw_square(im, 40, [0,0,255])
    draw_square(im, 30, [0,255,0])
    draw_square(im, 20, [255,0,0])
    draw_square(im, 10, [0,0,0])
    cv2.imwrite('im_2.png', im)



def draw_waves(im, pos, color):
    """Dibuja ondas en una posición y color dados
        :param im: (np.array) imagen en la que se dibuja
        :param pos: (int) posición en el eje y de las ondas
        :param color: (list or tuple) color de la onda
    """
    s =2
    x = np.array(range(W))
    y = np.sin(x)*s+pos

    y = y.astype(np.uint8)
    x = x.astype(np.uint8)

    im[list(y),list(x),:] = color

def im_3():
    im = np.zeros((W,H,3), np.uint8)
    for i in range(17):
        draw_waves(im, (i+2)*5, (255,i*15,100))
    
    cv2.imwrite('im_3.jpg', im)
    # show_im(im)




def show_im(im):
    cv2.imshow('window', im)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def main():
    im_1()
    # im_2()
    # im_3()


if __name__ == '__main__': main()