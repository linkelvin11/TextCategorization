import tc_bayes

training = input("input the name of the training file: ")
folds = int(input("enter number of folds to run "))

classy = tc_bayes.Classifier(0)

classy.kfold(training,folds)