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
  tPred = lr.predictClasses(xNew,w,b)

  # how right were we?
  error = lr.calcError(tPred,tNew)
  print "error = ",error

if __name__ == "__main__":
  main()
