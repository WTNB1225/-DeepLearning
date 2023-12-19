import numpy as np

def fucntion_1(x):
  return np.sum(x**2)

def _numerical_gradient_1d(f,x):
  h = 1e-4 #0.0001
  grad = np.zeros_like(x)
  
  for idx in range(x.size):
    tmp_val = x[idx]
    x[idx] = float(tmp_val) + h
    fxh1 = f(x)
    # print(fxh1)
    x[idx] = tmp_val - h
    fxh2 = f(x)
    #print(fxh2)
    #print(fxh1 - fxh2)
    #print((fxh1 - fxh2) / (2*h))
    grad[idx] = (fxh1 - fxh2) / (2*h)
    
    x[idx] = tmp_val
  #print(grad)
  return grad
    
def numerical_gradient_2d(f, X):
  if X.ndim == 1:
    return _numerical_gradient_1d(f,X)
  else:
    grad = np.zeros_like(X)
    
    for idx, x in enumerate(X):
      grad[idx] = _numerical_gradient_1d(f,x)
    
    return grad

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
    #print("------------------------------------")
    grad[idx] = (fxh1 - fxh2) / (h*2)
    x[idx] = tmp_val
    it.iternext()
  
  return grad
