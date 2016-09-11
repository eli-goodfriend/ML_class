"""
makes with the testing the stuff
"""
import logisticRegression as lr
from scipy import optimize
import numpy as np

def testGenerateData():
  testFails = False
  eps = 1.e-7 # single precision

  # check that generated data has the right labels based on set weights
  N = 20 # number of training data points
  M = 2  # dimension of data
  K = 3  # number of classes

  [x,t] = lr.generateData(N,M,K)
  w, b = lr.generateWeights(M,K)   

  lr.plotData(x, t, w, b, K)

  y = lr.predictClasses(x,w,b)
  shouldBeT = lr.setTFromY(y) 
  error = t - shouldBeT
  if (error.max() > eps):
    testFails = True
    print "error =", error
  return testFails

def testPullData():
  testFails = False
  
  # check that the real data has the expected range
  weightCost = 0.5 
  Ntrain = 10 # for each numeral
  Ntest = 5 # for each numeral
  [xTrain,tTrain, xTest, tTest, M, K] = lr.pullData("mnist_all.mat", Ntrain, Ntest)

  if ( (np.amax(xTrain) > 255) or
       (np.amin(xTrain) < 0) or
       (np.amax(xTest) > 255) or
       (np.amin(xTest) < 0) or
       (np.amax(tTrain) > 9) or
       (np.amin(tTrain) < 0) or
       (np.amax(tTest) > 9) or
       (np.amin(tTest) < 0) ):
    testFails = True
    print "Some of pulled data is out of expected range"

  return testFails

def testUseScipyOptimize():
  testFails = False
  eps = 1.e-7 # single precision
  
  # check that we're using the scipy optimize fcn correctly
  actualMin = 0.
  guess = 1.
  [minimum, error, info] = optimize.fmin_l_bfgs_b(testObjective, 
                                    guess, fprime=testGradObjective)
  if (abs(minimum - actualMin) > eps):
    testFails = True
    print "minimum =", actualMin, "guess = ", minimum 
  return testFails
def testObjective(w):
  return w*w
def testGradObjective(w):
  return 2*w

def testLogRegObjectiveOpt():
  testFails = False
  eps = 1.e-7 # single precision
  
  # check that the two log reg obj functions do the same thing
  N = 20 # number of training data points
  M = 2  # dimension of data
  K = 3  # number of classes
  weightCost = 0.5

  [x,t] = lr.generateData(N,M,K)
  w, b = lr.generateWeights(M,K)   
  wb = np.vstack((w,b))
  wb.shape = ((M+1)*K,)
  error = lr.logRegObjective(w, b, t, x, weightCost)
  errorOpt = lr.logRegObjectiveOpt(wb, t, x, weightCost, M, K)

  if (abs(error - errorOpt) > eps):
    testFails = True
    print "error =", error, "errorOpt =", errorOpt
  return testFails

def testLogRegGradOpt():
  testFails = False
  eps = 1.e-7 # single precision
  
  # check that the two log reg obj grad functions do the same thing
  N = 20 # number of training data points
  M = 2  # dimension of data
  K = 3  # number of classes
  weightCost = 0.5

  [x,t] = lr.generateData(N,M,K)
  w, b = lr.generateWeights(M,K)   
  wb = np.vstack((w,b))
  wb.shape = ((M+1)*K,)
  gradient = lr.logRegGrad(w, b, t, x, weightCost)
  gradientOpt = lr.logRegGradOpt(wb, t, x, weightCost, M, K)

  gradient.shape = ((M+1)*K)
  if (abs(gradient - gradientOpt).any() > eps):
    testFails = True
    print "gradient =", gradient
    print "gradientOpt =", gradientOpt
  return testFails

def testLogRegGrad():
  testFails = False
  eps = 1.e-7 * 100 # single precision with buffer, since this is numerics
  
  # check that the two log reg obj grad functions do the same thing
  N = 20 # number of training data points
  M = 2  # dimension of data
  K = 3  # number of classes
  weightCost = 0.5

  [x,t] = lr.generateData(N,M,K)
  w, b = lr.generateWeights(M,K)   
  wb = np.vstack((w,b))
  wb.shape = ((M+1)*K)
  error = optimize.check_grad(lr.logRegObjectiveOpt, lr.logRegGradOpt, wb, t, x, weightCost, M, K)
  if (error > eps):
    testFails = True
    print "error in gradient =", error
  return testFails

def main():
  tests = ['testGenerateData', 
           'testPullData',
           'testUseScipyOptimize',
           'testLogRegObjectiveOpt',
           'testLogRegGradOpt',
           'testLogRegGrad']

  for test in tests:
    testToCall = globals().copy().get(test)
    testFails = testToCall()
    if (testFails):
      print test, "failed"
    else:
      print test, "passed"

if __name__ == "__main__":
  main()
