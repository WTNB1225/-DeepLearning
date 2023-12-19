import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from functions.functions import *

def numerical_gradient(f,x):
  h = 1e-4
  grad = np.zeros_like(x)
  
  it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x)
    #print(fxh1)

    x[idx] = tmp_val - h
    fxh2 = f(x)
    #print(fxh2)
    #print((fxh1 - fxh2) / (h*2))
    #print("------------------------------------")
    grad[idx] = (fxh1 - fxh2) / (h*2)
    x[idx] = tmp_val
    it.iternext()
  
  return grad


class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2,3)
    
  def predict(self,x):
    return np.dot(x,self.W)
  
  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    #print(y)
    loss = cross_entropy_error(y, t)
    
    return loss
  
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
def f(w):
  return net.loss(x,t)
dW = numerical_gradient(f, net.W)
print(dW)