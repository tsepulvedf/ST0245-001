import os
from PIL import Image
import numpy as np

# Carpetas/listas
listaEnfermos = os.listdir("enfermo_csv")
listaSanos = os.listdir("sano_csv")
enfermos = "enfermo_csv/"
sanos = "sano_csv/"

# Recorrer listas y generar matrices + resizing
def estructuraEnfermos(enfermos, NombreArchivo):
    matriz = np.genfromtxt(enfermos + NombreArchivo, delimiter=",")
    print("Nombre del archivo: ", NombreArchivo)
    return matriz

def estructuraSanos(sanos, j):
    matriz = np.genfromtxt(sanos+j, delimiter=",")
    print("Nombre del archivo: ", j)
    return matriz

#Matrices escaladas/redimensionadas usando un valor de prueba arbitrario
def estructuraEnfermosResized(enfermos, NombreArchivo):
    matriz = np.genfromtxt(enfermos + NombreArchivo, delimiter=",")
    print("Nombre del archivo: ", NombreArchivo)
    imgObj = Image.fromarray(matriz)
    resized_imgObj = imgObj.resize((500, 780))
    resized_matriz = np.asarray(resized_imgObj)
    return resized_matriz


def estructuraSanosResized(sanos, j):
    matriz = np.genfromtxt(sanos+j, delimiter=",")
    print("Nombre del archivo: ", j)
    imgObj = Image.fromarray(matriz)
    resized_imgObj = imgObj.resize((224, 224))
    resized_matriz = np.asarray(resized_imgObj)
    return resized_matriz

#Imprimir funciones
#Tamaños originales:

for j in listaSanos:
    matriz1 = estructuraSanos(sanos, j)
    print("Tamaño/dimensiones de la imagen: ", len(matriz1), " x ", len(matriz1[0]))

for NombreArchivo in listaEnfermos:
    matriz2 = estructuraEnfermos(enfermos, NombreArchivo)
    print("Tamaño/dimensiones de la imagen: ", len(matriz2), " x ", len(matriz2[0]))

#Redimensionadas a valor elegido

for j in listaSanos:
    matriz1 = estructuraSanosResized(sanos, j)
    print("Tamaño/dimensiones de la imagen: ", len(matriz1), " x ", len(matriz1[0]))

for NombreArchivo in listaEnfermos:
    matriz2 = estructuraEnfermosResized(enfermos, NombreArchivo)
    print("Tamaño/dimensiones de la imagen: ", len(matriz2), " x ", len(matriz2[0]))
