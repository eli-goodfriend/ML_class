"""
makes with the testing the stuff
"""
import logisticRegression as lr

# test artificial data generation
def testGenerateData():
  testFails = False
  eps = 1.e-7 # single precision

  N = 20 # number of training data points
  M = 2  # dimension of data
  K = 3  # number of classes

  [x,t] = lr.generateData(N,M,K)

  # check that generated data has the right labels based on set weights
  w, b = lr.generateWeights(M,K)   
  y = lr.predictClasses(x,w,b)
  shouldBeT = lr.setTFromY(y) 
  error = t - shouldBeT
  if (error.max() > eps):
    testFails = True

  return testFails

# test loading of real MNIST data
def testPullData():
  return True

# test the objective function is correct
def testLogRegObjective():
  return True

# test the gradient of the objective function
def testLogRegGrad():
  return True

def main():
  tests = ['testGenerateData', 
           'testPullData',
           'testLogRegObjective',
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
