#!/usr/bin/python3

import os
import tc_bayes
import numpy as np

files = input("enter relative path to training set: ") or "corpus1_train.labels"
print("training based on provided training set:",files)
classy.get_training_files(files)
classy.get_wc()

files = input("enter relative path to test set: ") or "corpus1_test.list"
output = input("enter output filename: ") or "output.txt"
print("testing set:",files,"->",output)
classy.test(files, output)

print ("done")