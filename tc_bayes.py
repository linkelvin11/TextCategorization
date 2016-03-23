import numpy as np
import os
import nltk

class Classifier:

	def __init__ (self, verbose = 0):
		self.verbose = verbose

		self.labels = {}
		self.wc_by_doc = {}
		self.wc_by_class = {}
		
		self.nc = {}
		self.n = 0

		self.dictionary = []
		self.dict_size = 0

	# need Nc/N to get P(c)
	# cmap = argmax[ log P(c) + SUM{ log P(t|c)}]

	def get_training_files(self, corpus):
		wd = os.getcwd()
		corpus_root = wd + "/TC_provided/" + corpus
		training_files = corpus_root + "_train.labels"

		if self.verbose:
			print(os.path.abspath(training_files))

		for fname in open(training_files):
			docname = fname.split()[0]
			classname = fname.split()[1]
			self.labels[docname] = classname

			if self.verbose:
				print("file: "+docname+"\tlabel: "+classname)

			if classname not in self.nc:
				self.nc[classname] = 1
			else:
				self.nc[classname] += 1
			self.n += 1

			if classname not in self.wc_by_class:
				self.wc_by_class[classname] = {}

		return self.labels

	def get_wc(self):
		dir_root = os.getcwd() + "/TC_provided/"
		regex_tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z]+')
		for docname in self.labels:
			classname = self.labels[docname]

			if self.verbose:
				print("training on"+docname)
			if docname not in self.wc_by_doc:
				self.wc_by_doc[docname] = {}

			with open(os.path.join(dir_root, docname),'r') as doc:
				text = doc.read().lower()
				tokens = regex_tokenizer.tokenize(text)
				for token in tokens:

					if token not in self.dictionary:
						self.dictionary.append(token)
						self.dict_size += 1

					if token not in self.wc_by_doc[docname]:
						self.wc_by_doc[docname][token] = 1
					else:
						self.wc_by_doc[docname][token] += 1

					if token not in self.wc_by_class[classname]:
						self.wc_by_class[classname][token] = 1
					else:
						self.wc_by_class[classname][token] += 1
		return

	def get_priors(self):


		return




	
	