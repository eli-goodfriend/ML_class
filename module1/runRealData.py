"""
makes with the running the stuff
"""
import logisticRegression as lr

def main():
  weightCost = 0.5 
  Ntrain = 20 # for each numeral
  Ntest = 5 # for each numeral

  # get the training and testing data from the MNIST dataset
  [xTrain,tTrain, xTest, tTest, M, K] = lr.pullData("mnist_all.mat", Ntrain, Ntest)

  # train a model to classify the data
  [w,b] = lr.getWeights(xTrain, tTrain, Ntrain*10, M, K, weightCost)

  # classify the test data using the model
  yPred = lr.predictClasses(xTest, w, b)
  tPred = lr.setTFromY(yPred)

  # how right were we?
  error = lr.calcError(yPred,tTest)
  print "SSE (yPred - t) = ",error
  numWrong = lr.numWrong(tPred,tTest)
  print numWrong, "incorrect out of", Ntest*10

if __name__ == "__main__":
  main()
