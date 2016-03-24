#!/usr/bin/python3

import os
import tc_bayes
import numpy as np

classy = tc_bayes.Classifier(0)
files = input("enter relative path to training set: ")
print("training based on provided training set:",files)
classy.get_training_files(files)
classy.get_wc()

files = input("enter relative path to test set: ")
print("testing set:",files)
classy.test(files)

print ("done")