import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

def getWeights(x, t, N, M, K, weightCost):
  w0 = np.ones((M+1)*K)
  [weights, error, info] = optimize.fmin_l_bfgs_b(logRegObjectiveOpt, w0, args=(t, x, weightCost, M, K), fprime=logRegGradOpt) 
  weights.shape = (M+1,K)
  [w,b] = np.vsplit(weights, [M])
  return w, b

def logRegObjectiveOpt(wb, t, x, weightCost, M, K):
  [w,b] = np.split(wb, [K*M])
  w.shape = (M,K)
  b.shape = (1,K)
  error = logRegObjective(w, b, t, x, weightCost)
  return error

def logRegObjective(w, b, t, x, weightCost):
# TODO this is janky
  [N,K] = t.shape
  M = w.size / K
  error = 0
  y = np.zeros((N,K))
  y = predictClasses(x, w, b)
  for i in range(N):
    for j in range(K):
      error += -t[i,j]*np.log(y[i,j])
  for i in range(M):
    for j in range(K):
      error += weightCost*w[i,j]**2
  return error

def logRegGradOpt(wb, t, x, weightCost, M, K):
  [w,b] = np.split(wb, [K*M])
  w.shape = (M,K)
  b.shape = (1,K)
  grad = logRegGrad(w, b, t, x, weightCost)
  return grad

def logRegGrad(w, b, t, x, weightCost):
  [N,M] = x.shape
  [one,K] = b.shape
  grad = np.zeros((M+1,K))
  y = np.zeros((N,K))
  y = predictClasses(x, w, b)
  for k in range(K):
    for n in range(N):
      for m in range(M):
        grad[m,k] += (y[n,k] - t[n,k])*x[n,m]  
      grad[M,k] += (y[n,k] - t[n,k])*1. # for bias
  grad.shape = ((M+1)*K)
  return grad

def predictClasses(x,w,b):
  [N,M] = x.shape
  a = np.dot(x,w) + np.dot(np.ones((N,1)),b)
  [N,K] = a.shape
  yPred = np.zeros((N,K))
  for n in range(N):
    yPred[n,:] = softmax(a[n,:])
  return yPred

def generateData(N,M,K):
  x = np.random.random((N,M))*2. - 1.

  w = np.vstack((np.ones(K), np.arange(K)*2-1)) + np.random.normal(scale=0.05, size=(M,K))
  b = np.arange(K)*0.5 + np.random.normal(scale=0.05, size=(1,K))

  y = predictClasses(x,w,b)
  t = np.zeros((N,K))
  for rowIdx in range(N):
    maxIdx = np.argmax(y[rowIdx,:])
    t[rowIdx,maxIdx] = 1

  plotData(x,t,w,b,K)
  return x, t

def plotData(x,t,w,b,K):
  [x1, x2] = np.hsplit(x, [1])
  label = np.dot(t, np.arange(K)+1)
  
  plt.scatter(x1, x2, c=label, alpha=0.5, s=75)
  for idx in range(K):
    line = -w[0,idx]/w[1,idx] * x1 - b[0,idx]/w[1,idx]
    plt.plot(x1,line)
  plt.show()

def calcError(tPred,tReal):
  error = (tPred - tReal)**2
  return np.sum(error)

def softmax(a):
  K = a.size
  softmax = np.zeros((K))
  maxval = np.amax(a) # TODO incorrect, want largest magnitude
  a -= maxval # for stability
  denom = 0
  for colIdx in range(K):
    denom += np.exp(a[colIdx])
    softmax[colIdx] = np.exp(a[colIdx])
  softmax /= denom 
  return softmax
