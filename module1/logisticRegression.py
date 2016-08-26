import numpy as np
from matplotlib import pyplot as plt

def getWeights(x,t,N,M,K):
  w = np.ones((M,K))
  return w

def predictClasses(x,w,N,M,K):
  t = np.zeros((1,N))
  return t

def generateData(N,M,K):
  x1 = np.random.random(N)
  x2 = np.random.random(N)
  x = np.vstack((x1,x2))
  
  w = np.ones((M,K))
  wT = w.transpose()
  b = np.ones((K,N)) * -0.5
  y = np.dot(wT,x) + b
  t = np.zeros((1,N))
  
  # if y>0, then we're in class 1, otherwise in class 0
  class1 = y>0
  t += class1 * 1

  ## write it out 
  #writeData = np.vstack((x,t))
  #np.savetxt('synData',writeData)

  plotData(x,t)

  return x, t

def plotData(x,t):
  
  #dataIn = np.loadtxt('synData')
  
  #[x, t] = np.vsplit(dataIn, [M])
  [x1, x2] = np.vsplit(x, [1])
  
  plt.scatter(x1, x2, c=t, alpha=0.5, s=75)
  plt.show()

def calcError(tPred,tReal):
  return 0
