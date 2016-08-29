import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

def getWeights(x, t, N, M, K, weightCost):
  w0 = np.ones((M+1,K))
  [weights, error, info]  = optimize.fmin_l_bfgs_b(logRegObjectiveOpt, w0, args=(t, x, weightCost, M, K), approx_grad=True) 
  weights.shape = (M+1,K)
  [w,b] = np.vsplit(weights, [M])
  return w, b

def logRegObjectiveOpt(wb, t, x, weightCost, M, K):
  [w,b] = np.split(wb, [K*M])
  w.shape = (M,K)
  error = logRegObjective(w, b, t, x, weightCost)
  return error

def logRegObjective(w, b, t, x, weightCost):
# TODO this is janky
  [N,K] = t.shape
  M = w.size / K
  error = 0
  for i in range(N):
    y = predictClasses(x[i,:], w, b)
    for j in range(K):
      error += -t[i,j]*np.log(y[j])
  for i in range(M):
    for j in range(K):
      error += weightCost*w[i,j]**2
  return error

def logRegGrad(w, b, t, x, weightCost):
  return t

def predictClasses(x,w,b):
  a = np.dot(x,w) + b
  #TODO this is janky
  if a.ndim > 1:
    [N,K] = a.shape
    yPred = np.zeros((N,K))
    for n in range(N):
      yPred[n,:] = softmax(a[n,:])
  else:  
    K = a.size
    yPred = np.zeros((K))
    yPred = softmax(a)
  return yPred

def generateData(N,M,K):
  x = np.random.random((N,M))
  w = np.random.random((M,K))
  b = np.ones((1,K)) * -0.5

  y = predictClasses(x,w,b)
  t = np.zeros((N,K))
  for rowIdx in range(N):
    maxIdx = np.argmax(y[rowIdx,:])
    t[rowIdx,maxIdx] = 1

  plotData(x,t,K)
  return x, t

def plotData(x,t,K):
  [x1, x2] = np.hsplit(x, [1])
  label = np.dot(t, np.arange(K))
  
  plt.scatter(x1, x2, c=label, alpha=0.5, s=75)
  plt.show()

def calcError(tPred,tReal):
  error = (tPred - tReal)**2
  return np.sum(error)

def softmax(a):
  # TODO this are numerically badz
  K = a.size
  softmax = np.zeros((K))
  denom = 0
  for colIdx in range(K):
    denom += np.exp(a[colIdx])
    softmax[colIdx] = np.exp(a[colIdx])
  softmax /= denom 
  return softmax
