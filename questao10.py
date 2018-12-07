#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as mp
import numpy as np
import math

def toRBG(image):
	return image[:,:,0:3]

def nchannels(imagem):
	dimensoes = imagem.shape
	ultimo = len(dimensoes) - 1
	return dimensoes[ultimo]


def imread(nomeArquivo):	
	imagem = mp.imread(nomeArquivo)
	if imagem.dtype == 'float32':
		imagem.dtype = np.uint8
		imagem * 255
	if nchannels(imagem) > 3: #trata caso em que imagens tem mais que 3 canais
		return toRBG(imagem)
	return imagem


def size(imagem):
	dimensoes = imagem.shape
	return np.array(dimensoes[0:2])





def rgb2gray(image):
    image.setflags(write=1)
    tam = size(image)
    x = tam[0]
    y = tam[1]
    print(x)
    print(y)
    result = np.ndarray(shape=[x,y])
    for row in range(0,x):
        for col in range(0,y):
            mul = image[row][col] #[[0.299],[0.587],[0.1114],[0]]
            
            mul[0] = mul[0]*0.299
            mul[1] = mul[1]*0.587
            mul[2] = mul[2]*0.1114


            result[row][col] = sum(mul)
    return result


def imreadgray(name):
	img = imread(name) # read the image
	if(nchannels(img) > 1):
		return rgb2gray(img) 
	return img

def imshow(image):
	if(nchannels(image) == 1):
		mp.imshow(image, cmap = 'gray', interpolation = 'nearest')
	else:
		mp.imshow(image,cmap = 'gray')
	
	mp.show()

def thresh(image, limiar):
	dims = size(image)
	x = dims[0]
	y = dims[1]
	#print(image)
	img = np.ndarray(shape=[x,y,3])
	
	for linha in range(0,x):
		for coluna in range(0,y):
			l = [255 if y >= limiar else 0 for y in image[linha][coluna]]
			#lim[0] = 255 if lim[0] >= limiar else 0
			#lim[1] = 255 if lim[1] >= limiar else 0
			#lim[2] = 255 if lim[2] >= limiar else 0
			#img[linha][coluna] = lim
			print(l)
			img[linha][coluna] = l
			
	return img
	return ((image >= limiar) * l).astype(np.uint8)

def negative(image):
	dims = size(image)
	x = dims[0]
	y = dims[1]
	#print(image)
	img = np.ndarray(shape=[x,y,3])
	
	for linha in range(0,x):
		for coluna in range(0,y):
			img[linha][coluna] = [abs(pixel - 255) for pixel in image[linha][coluna]]
	

	return img
	L = 255
	return (L - image).astype(np.uint8)

#func 10

def contrast(img, r, m):
    newImg = img.copy()
    if (nchannels(img) == 1): 
        for x in range(0, len(newImg)): 
            for y in range(0, len(newImg[0])):
                tmp = r*(img[x][y]-m)+m
                if (tmp >= 255):
                    newImg[x][y]  = 255
                elif (tmp <= 0):
                    newImg[x][y] = 0
    else: 
        for x in range(0, len(newImg)): 
            for y in range(0, len(newImg[0])):
                tmp = r*(img[x][y][0]-m)+m
                if (tmp >= 255):
                    newImg[x][y][0]  = 255
                elif (tmp <= 0):
                    newImg[x][y][0] = 0
                tmp = r*(img[x][y][1]-m)+m
                if (tmp >= 255):
                    newImg[x][y][1]  = 255
                elif (tmp <= 0):
                    newImg[x][y][1] = 0
                    tmp= r*(img[x][y][2]-m)+m
                if (tmp  >= 255):
                    newImg[x][y][2]  = 255
                elif (tmp <= 0):
                    newImg[x][y][2] = 0
    return newImg


contrast(imread("exemplo.jpg"),2.2,3.5)


# In[ ]:




