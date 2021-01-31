import numpy as np
import math

def sigmoid(x):
    
    s = 1/(1+np.exp(-x))
    
    return s

def sigmoidderivative(x):

    s = sigmoid(x)
    ds = s*(1-s)
    
    return ds
    
def image2vector(image):

    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    
    return v
    
def rownormalizer(x):
    x_norm = np.linalg.norm(x,ord=#,axis=#,keepdims=True)
    
    x = x/x_norm

    return x
    
def softmax(x):

    x_exp = np.exp(x)

    x_sum = np.sum(x_exp,axis=#,keepdims=True)
    
    s = x_exp/x_sum

    
    return s
