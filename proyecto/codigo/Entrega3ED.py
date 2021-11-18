#Aplicacion de huffman adaptada de https://github.com/DexteR891161/Image_Compression-Huffman_Coding-/blob/master/app.py

import os
from PIL import Image
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import scipy.ndimage
import cv2

# Carpetas/listas
listaEnfermos = os.listdir("enfermo_csv")
listaEnfermosCOMP = os.listdir("enfermoComprimido_csv")
listaSanos = os.listdir("sano_csv")
listaSanosCOMP = os.listdir("sanoComprimido_csv")
enfermos = "enfermo_csv/"
enfermosCOMP = "enfermoComprimido_csv/"
sanos = "sano_csv/"
sanosCOMP = "sanoComprimido_csv/"

# Recorrer listas y generar matrices
def estructuraEnfermos(enfermos, NombreArchivo):
    matriz = np.genfromtxt(enfermos + NombreArchivo, delimiter=",")
    print("Nombre del archivo: ", NombreArchivo)
    return matriz

def estructuraSanos(sanos, j):
    matriz = np.genfromtxt(sanos+j, delimiter=",")
    print("Nombre del archivo: ", j)
    return matriz

#Compresion con perdidas (Bilinear interpolation)


for NombreArchivo in listaEnfermos:
    matriz1 = estructuraEnfermos(enfermos, NombreArchivo)
    imgObj = scipy.ndimage.zoom(matriz1, 0.3, order=1)
    print("Tamaño/dimensiones originales de la imagen: ", len(matriz1), " x ", len(matriz1[0]))
    if NombreArchivo.endswith(".csv"):
        f = genfromtxt("enfermo_csv/"+NombreArchivo, delimiter=",")  # Convierten el csv a array de numpy
        #plt.imshow(imgObj, cmap="gray")
        #plt.axis('off')        #Muestra la imagen
        #plt.show()
        np.savetxt("enfermoComprimido_csv/"+NombreArchivo, imgObj,fmt='%s', delimiter=",")



for j in listaSanos:
    matriz1 = estructuraSanos(sanos, j)
    imgObj = scipy.ndimage.zoom(matriz1, 0.3, order=1)
    print("Tamaño/dimensiones originales de la imagen: ", len(matriz1), " x ", len(matriz1[0]))
    if j.endswith(".csv"):
        f = genfromtxt("sano_csv/"+j, delimiter=",")  # Convierten el csv a array de numpy
        plt.imshow(imgObj, cmap="gray")
        plt.axis('off')
        np.savetxt("sanoComprimido_csv/"+j, imgObj,fmt='%s', delimiter=",")
        #plt.show()

#Descompresion con perdidas (Bilinear interpolation)

for j in listaSanosCOMP:
    matriz1 = estructuraSanos(sanosCOMP, j)
    imgObj = scipy.ndimage.zoom(matriz1, 1.2, order=1) #Forma de compresion y multiplicamos o dividimos su escalamiento
    print("Tamaño/dimensiones originales de la imagen: ", len(matriz1), " x ", len(matriz1[0]))
    if j.endswith(".csv"):
        f = genfromtxt(sanosCOMP+j, delimiter=",")  # Convierten el csv a array de numpy
        #plt.imshow(imgObj, cmap="gray") Muestra las imagenes
        #plt.axis('off')
        #plt.show()
        np.savetxt("sanoDescomprimido_csv/" + j, imgObj, fmt='%s', delimiter=",")



for NombreArchivo in listaEnfermosCOMP:
    matriz1 = estructuraEnfermos(enfermosCOMP, NombreArchivo)
    imgObj = scipy.ndimage.zoom(matriz1, 1.2, order=1)
    print("Tamaño/dimensiones originales de la imagen: ", len(matriz1), " x ", len(matriz1[0]))
    if NombreArchivo.endswith(".csv"):
        f = genfromtxt(enfermosCOMP+NombreArchivo, delimiter=",")  # Convierten el csv a array de numpy
        #plt.imshow(imgObj, cmap="gray")
        #plt.axis('off')        #Muestra la imagen
        #plt.show()
        np.savetxt("enfermoDescomprimido_csv/"+NombreArchivo, imgObj,fmt='%s', delimiter=",")



#Se crean los arboles de huffman y la codificacion y descodificacion del mismo algoritmo
class Node:
    def __init__(self, data, freq, left=None, right=None):
        self.data = data
        self.freq = freq
        self.left = left
        self.right = right

class Huffman:
    def __init__(self, img):
        self.dict = {}
        self.code = []
        self.hist_dic = {}
        self.img = img
        hist, bins = np.histogram(self.img.ravel(), 256, [0, 256])
        bins = bins.tolist()
        hist = hist.tolist()
        for hist, bin in zip(hist, bins):
            self.hist_dic[bin]=hist
        dic = self.hist_dic.copy()
        for key in dic.keys():
            if self.hist_dic[key] == 0:
                del(self.hist_dic[key])

        self.bins = self.hist_dic.keys()
        self.hist = [self.hist_dic[x] for x in self.bins]

    def CrearArbol(self):
        charList = self.bins
        freqList = self.hist
        minHeap = [Node(c, f) for c, f in zip(charList, freqList)]

        while(len(minHeap)!=1):
            minHeap = sorted(minHeap, key=lambda x:x.freq, reverse=True)
            intNode = Node(None, minHeap[-1].freq+minHeap[-2].freq)
            intNode.left = minHeap[-2]
            intNode.right = minHeap[-1]
            minHeap.pop()
            minHeap.pop()
            minHeap.append(intNode)

        return minHeap[0]

    def Code(self, tree, s=""):
        if tree.data is not None:
            print(tree.data, end=" ")
            print(s)
            self.dict[tree.data] = s
            return
        self.Code(tree.left, s+'0')
        self.Code(tree.right, s+'1')

    def codificar(self):
        flat=self.img.flatten().tolist()
        for pix in flat:
            self.code.append(self.dict[pix])
        return self.code

    def decodificar(self, tree):
        current = tree
        self.string = []
        for code in self.code:
            for bit in code:
                if bit == "0":
                    current = current.left
                else:
                    current = current.right

            if current.left is None and current.right is None:
                self.string.append(current.data)
                current = tree
        return self.string

#Loop para la compresion de huffman de las imagenes

for j in listaSanosCOMP:
    image = cv2.imread(sanosCOMP+j, 0)
    objeto = Huffman(image)
    raiz = objeto.CrearArbol()
    objeto.Code(raiz)
    coded_image = objeto.codificar()

    #cv2.imshow("Imagen original", image) Se muestra la imagen
    cv2.waitKey()

    shape = image.shape

    ret = objeto.decodificar(raiz)
    ret = np.array(ret, np.uint8)
    ret_image = np.reshape(ret, shape)
    #cv2.imshow("Tras compresion", ret_image) Se muestra la imagen
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("sanoComprimido2Algoritmos_csv/" + j, image) #Se guardan las imagenes tras la compresion


for NombreArchivo in listaEnfermosCOMP:
    image = cv2.imread(enfermosCOMP+NombreArchivo, 0)
    objeto = Huffman(image)
    raiz = objeto.CrearArbol()
    objeto.Code(raiz)
    coded_image = objeto.codificar()

    #cv2.imshow("Imagen original", image) Se muestra la imagen
    #cv2.waitKey()

    shape = image.shape

    ret = objeto.decodificar(raiz)
    ret = np.array(ret, np.uint8)
    ret_image = np.reshape(ret, shape)
    #cv2.imshow("Tras compresion", ret_image) Se muestra la imagen
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    cv2.imwrite("enfermoComprimido2Algoritmos_csv/" + NombreArchivo, image) #Se guardan las imagenes tras la compresion

