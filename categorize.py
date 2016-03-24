#!/usr/bin/python3

import os
import tc_bayes
import numpy as np

classy = tc_bayes.Classifier(0)
classy.get_training_files("corpus1")

classy.get_wc()

classy.test()

print ("done")