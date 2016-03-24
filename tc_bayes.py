import numpy as np
import os
import nltk
import math

class Classifier:

	def __init__ (self, verbose = 0):
		self.verbose = verbose

		self.labels = {}
		self.wc_by_doc = {}
		self.wc_by_class = {}
		self.wc = {}
		
		self.nc = {}
		self.n = 0

		self.dictionary = {}
		self.dict_size = 0

		self.regex = '[a-zA-z]+'

		self.stop_words = {}
		for word in open("stop_words.txt",'r'):
			self.stop_words[word.rstrip()] = 1

		return

	def get_training_files(self, corpus):
		wd = os.getcwd() + "/"
		training_files = wd + corpus

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
		dir_root = os.getcwd() + "/"
		regex_tokenizer = nltk.tokenize.RegexpTokenizer(self.regex)
		for docname in self.labels:
			classname = self.labels[docname]

			if self.verbose:
				print("training on"+docname)
			if docname not in self.wc_by_doc:
				self.wc_by_doc[docname] = {}
			if classname not in self.wc_by_class:
				self.wc_by_class[classname] = {}

			with open(os.path.join(dir_root, docname),'r') as doc:
				text = doc.read().lower()
				tokens = regex_tokenizer.tokenize(text)
				for token in tokens:

					if token in self.stop_words:
						continue

					if token not in self.dictionary:
						self.dictionary[token] = 1
						self.dict_size += 1

					if token not in self.wc_by_doc[docname]:
						self.wc_by_doc[docname][token] = 1
					else:
						self.wc_by_doc[docname][token] += 1

					if token not in self.wc_by_class[classname]:
						self.wc_by_class[classname][token] = 1
					else:
						self.wc_by_class[classname][token] += 1

					if classname not in self.wc:
						self.wc[classname] = 1
					else:
						self.wc[classname] += 1
		return

	def test(self, corpus):
		dir_root = os.getcwd() + "/"
		test_files = dir_root + corpus

		output = open("output.txt",'w')

		for fname in open(test_files):
			fname = fname.rstrip()

			with open(os.path.join(dir_root,fname),'r') as doc:
				max_prob = -99999999
				prior = {}
				max_class = "asdf"
				text = doc.read().lower()
				regex_tokenizer = nltk.tokenize.RegexpTokenizer(self.regex)
				tokens = regex_tokenizer.tokenize(text)

				for curr_class in self.wc_by_class:
					curr_prob = np.log(self.nc[curr_class]) - np.log(self.n)
					for token in tokens:
						if token in self.stop_words:
							continue

						if token not in self.wc_by_class[curr_class]:
							curr_prob -= np.log(self.wc[curr_class] + self.dict_size)
						else:
							curr_prob += np.log(self.wc_by_class[curr_class][token] + 1)
							curr_prob -= np.log(self.wc[curr_class] + self.dict_size)

					if curr_prob > max_prob:
						max_prob = curr_prob
						max_class = curr_class
					if self.verbose:
						print(curr_class,curr_prob)
				if self.verbose:
					print ("max:",max_class,curr_prob)
				output.write(fname+" "+max_class+"\n")
		return