"""
makes with the running the stuff
"""
import logisticRegression as lr

def main():
  N = 15 # number of training data points
  M = 2  # dimension of data
  K = 3  # number of classes
  weightCost = 0.5 # punishment for large weights

  # get the training data, either generated or real
  [x,t] = lr.generateData(N,M,K)

  # train a model to classify the data
  [w,b] = lr.getWeights(x,t,N,M,K,weightCost)

  # get new data
  [xNew,tNew] = lr.generateData(N,M,K)

  # classify the new data using the model
  tPred = lr.predictClasses(xNew,w,b)

  # how right were we?
  error = lr.calcError(tPred,tNew)
  print "error = ",error

if __name__ == "__main__":
  main()
