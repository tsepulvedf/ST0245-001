import os

#Carpetas/listas
listaEnfermos = os.listdir("enfermo_csv")
listaSanos = os.listdir("sano_csv")
enfermos = "enfermo_csv/"
sanos = "sano_csv/"

#Recorrer listas y generar matrices
def estructuraEnfermos(enfermos, NombreArchivo):
    imagenesEnfermos = open(enfermos+NombreArchivo)
    lector = imagenesEnfermos.readlines()
    print("Nombre del archivo: ",NombreArchivo)
    matriz = []
    for a in lector:
        matriz.append(a.split(","))
    return matriz      #Aquí se guarda la matriz

def estructuraSanos(sanos, j):
    imagenesSanos = open(sanos+j)
    lector = imagenesSanos.readlines()
    print("Nombre del archivo: ",j)
    matriz = []
    for k in lector:
        matriz.append(k.split(","))
    return matriz       #Aquí se guarda la matriz

#Imprimir funciones
for j in listaSanos:
    matriz1 = estructuraSanos(sanos, j)
    print("Tamaño/dimensiones de la imagen: " , len(matriz1) , " x " , len(matriz1[0]))

for NombreArchivo in listaEnfermos:
    matriz2 = estructuraEnfermos(enfermos, NombreArchivo)
    print("Tamaño/dimensiones de la imagen: " , len(matriz2) , " x " , len(matriz2[0]))

