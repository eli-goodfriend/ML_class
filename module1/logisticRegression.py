import numpy as np
from scipy import optimize
from scipy import io
from matplotlib import pyplot as plt

def getWeightsScipy(x, t, N, M, K, weightCost):
  w0 = np.ones((M+1)*K)
  [weights, error, info] = optimize.fmin_l_bfgs_b(logRegObjectiveOpt, w0, args=(t, x, weightCost, M, K), fprime=logRegGradOpt) 
  weights.shape = (M+1,K)
  [w,b] = np.vsplit(weights, [M])
  return w, b

def getWeights(x, t, N, M, K, weightCost):
  w0 = np.ones((M,K))
  b0 = np.zeros((1,K))
  [w, b] = stochasticGradientDescent(logRegObjective, logRegGrad, w0, b0, t, x, weightCost, M, K)
  return w, b

def logRegObjectiveOpt(wb, t, x, weightCost, M, K):
  [w,b] = np.split(wb, [K*M])
  w.shape = (M,K)
  b.shape = (1,K)
  error = logRegObjective(w, b, t, x, weightCost)
  return error

def logRegObjective(w, b, t, x, weightCost):
  eps = 1.e-8
  [N,K] = t.shape
  M = w.size / K
  error = 0
  y = predictClasses(x, w, b)

  if (np.amax(y) > 1.-eps): # single class chosen
    return error

  error = np.sum(-t*np.log(y))
  error += weightCost*np.sum(w**2)
  return error

def logRegGradOpt(wb, t, x, weightCost, M, K):
  [w,b] = np.split(wb, [K*M])
  w.shape = (M,K)
  b.shape = (1,K)
  grad = logRegGrad(w, b, t, x, weightCost)
  grad.shape = ((M+1)*K)
  return grad

def logRegGrad(w, b, t, x, weightCost):
  [N,M] = x.shape
  [one,K] = b.shape
  y = predictClasses(x, w, b)
  errSignal = y - t
  paddedx = np.hstack((x, np.ones((N,1))))
  grad = np.dot(paddedx.T, errSignal) + 2.*weightCost*np.vstack((w, np.zeros((1,K))))
  return grad

def predictClasses(x,w,b):
  [N,M] = x.shape
  a = np.dot(x,w) + np.dot(np.ones((N,1)),b)
  [N,K] = a.shape
  yPred = np.empty((N,K))
  for n in range(N):
    yPred[n,:] = softmax(a[n,:])
  return yPred

def generateWeights(M,K):
  w = np.ones((1,K))
  for dimension in range(M-1):
    w = np.vstack((w, np.arange(K)*2-1))
  #w += np.random.normal(scale=0.05, size=(M,K))
  b = np.arange(K)*0.5
  b.shape = (1,K)
  #b += np.random.normal(scale=0.05, size=(1,K))
  return w, b

def setTFromY(y):
  [N,K] = y.shape
  t = np.zeros((N,K))
  for rowIdx in range(N):
    maxIdx = np.argmax(y[rowIdx,:])
    t[rowIdx,maxIdx] = 1
  return t

def generateData(N,M,K):
  x = np.random.random((N,M))*2. - 1.
  w, b = generateWeights(M,K)
  y = predictClasses(x,w,b)
  t = setTFromY(y)
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

def numWrong(tPred, tReal):
  numWrong = np.count_nonzero(tPred - tReal)
  numWrong /= 2 # since one wrong t value will be different in 2 places
  return numWrong

def softmax(a):
  expA = np.exp(a - a.max())
  expSum = expA.sum()
  if (expSum==0):
    raise NameError('All classes have likelihood 0.')
  softmax = expA / expSum
  return softmax

def pullData(filename, Ntrain, Ntest):
  data = io.loadmat(filename) 
  [en,M] = data['train0'].shape
  K = 10 # there are ten numerals, or could read this from data
  xTrain = np.empty((Ntrain*K,M))
  tTrain = np.zeros((Ntrain*K,K))
  xTest = np.empty((Ntest*K,M))
  tTest = np.zeros((Ntest*K,K))
  for key in list(data.keys()): 
    if key[0:5]=="train":
      classLabel = int(key[5])
      xTrain[classLabel*Ntrain:(classLabel+1)*Ntrain, :] = data[key][0:Ntrain, :]/255.
      tTrain[classLabel*Ntrain:(classLabel+1)*Ntrain, classLabel] = 1
    elif key[0:4]=="test":
      classLabel = int(key[4])
      xTest[classLabel*Ntest:(classLabel+1)*Ntest, :] = data[key][0:Ntest, :]/255.
      tTest[classLabel*Ntest:(classLabel+1)*Ntest, classLabel] = 1

  return xTrain, tTrain, xTest, tTest, M, K

def gradientDescent(objective, gradObjective, w0, b0, t, x, weightCost, M, K, momentum = 0., dw = 0., db = 0., maxIter = 100, learnRate = 0.05, mbSize = 1):
  eps = 1.e-7
  w = w0
  b = b0
  oldObjVal = objective(w, b, t, x, weightCost)
  
  for iteration in range(maxIter): 
    objGrad = gradObjective(w, b, t, x, len(x)*weightCost)
    dw *= momentum
    db *= momentum
    dw -= learnRate*objGrad[0:M, :] / float(len(x))
    db -= learnRate*objGrad[M, :] / float(len(x))
    w += dw
    b += db
    newObjVal = objective(w, b, t, x, weightCost)
    diff = abs(newObjVal - oldObjVal)
    if (diff < eps):
      return w, b, dw, db, True
    else:
      oldObjVal = newObjVal
  return w, b, dw, db, False

def stochasticGradientDescent(objective, gradObjective, w0, b0, t, x, weightCost, M, K, momentum = 0.95, steps = 50000, learnRate = 0.05, mbSize = 64):
  eps = 1.e-7
  [N, em] = x.shape
  dataPts = range(N)
  dw = np.zeros((M,K))
  db = np.zeros((1,K))

  for step in range(steps):
    np.random.shuffle(dataPts)
    dataThisTime = dataPts[0:mbSize]
    tMB = t[dataThisTime, :]
    xMB = x[dataThisTime, :] 
    w, b, dw, db, done = gradientDescent(objective, gradObjective, w0, b0, tMB, xMB, weightCost, M, K, momentum, dw, db, 1, learnRate, mbSize)
    if (done):
      return w, b
    else:
      w0 = w
      b0 = b

  return w, b











