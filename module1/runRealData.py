"""
makes with the running the stuff
"""
import logisticRegression as lr

def main():
  weightCost = 0.5 
  Ntrain = 10 # for each numeral
  Ntest = 5 # for each numeral

  # get the training and testing data from the MNIST dataset
  [xTrain,tTrain, xTest, tTest, M, K] = lr.pullData("mnist_all.mat", Ntrain, Ntest)

  # train a model to classify the data
  [w,b] = lr.getWeights(xTrain, tTrain, Ntrain, M, K, weightCost)

  # get new data
  #[xNew,tNew] = lr.generateData(Ntest,M,K)

  # classify the new data using the model
  #tPred = lr.predictClasses(xNew,w,b)

  # how right were we?
  #error = lr.calcError(tPred,tNew)
  #print "error = ",error

if __name__ == "__main__":
  main()
