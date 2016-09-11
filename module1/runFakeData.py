"""
makes with the running the stuff
"""
import logisticRegression as lr

def main():
  Ntrain = 100 # number of training data points
  Ntest = 25 # number of test data points
  M = 2  # dimension of data
  K = 3  # number of classes
  weightCost = 0.5 # punishment for large weights

  # generate synthetic training data
  [x,t] = lr.generateData(Ntrain,M,K)

  # train a model to classify the data
  [w,b] = lr.getWeights(x,t,Ntrain,M,K,weightCost)

  # generate synthetic testing data
  [xNew,tNew] = lr.generateData(Ntest,M,K)

  # classify the new data using the model
  yPred = lr.predictClasses(xNew,w,b)
  tPred = lr.setTFromY(yPred)

  # how right were we?
  error = lr.calcError(yPred,tNew)
  print "SSE (yPred - t) = ",error
  numWrong = lr.numWrong(tPred,tNew)
  print numWrong, "incorrect out of", Ntest

if __name__ == "__main__":
  main()
