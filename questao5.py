#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as mp
import numpy as np
import math

#func 1
def imread(nomeArquivo):	
	imagem = mp.imread(nomeArquivo)
	if imagem.dtype == 'float32':
		imagem.dtype = np.uint8
		imagem * 255
	'''if nchannels(imagem) > 3: #trata caso em que imagens tem mais que 3 canais
		return toRBG(imagem)'''
	return imagem

def size(imagem):
	dimensoes = imagem.shape
	return np.array(dimensoes[0:2])


def rgb2gray(image):
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

img = imread("exemplo.jpg")
# img.flags
img.setflags(write=1)
#img.flags
rgb2gray(img)
