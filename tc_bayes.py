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

		self.regex = '[a-zA-z\.]+'

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
			self.get_training_file(docname, classname)

		return self.labels

	def get_training_file(self, docname, classname):
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

		return

	def get_wc(self):
		regex_tokenizer = nltk.tokenize.RegexpTokenizer(self.regex)
		for docname in self.labels:
			classname = self.labels[docname]

			if self.verbose:
				print("training on"+docname,classname)
			if docname not in self.wc_by_doc:
				self.wc_by_doc[docname] = {}
			if classname not in self.wc_by_class:
				self.wc_by_class[classname] = {}

			with open(os.path.join(os.getcwd(), docname),'r') as doc:
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

	def test(self, corpus, output="output.txt"):
		test_files = os.path.join(os.getcwd(),corpus)

		output = open(output,'w')

		for fname in open(test_files):
			fname = fname.split[0]
			max_class = self.test_file(fname)
			output.write(fname+" "+max_class+"\n")
		return

	def test_file(self, fname):
		path = os.path.join(os.getcwd(),fname)
		with open(path,'r') as doc:
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

		return max_class

	def kfold(self, corpus, k = 2):
		acc = 0
		wd = os.getcwd() + "/"
		training_files = wd + corpus

		with open(training_files) as f:
			files = f.readlines()
		files = np.array_split(files,k)

		for fold in range(k):
			correct = 0
			total = 0
			self.__init__(0)
			for i in range(k):
				if i == fold:
					test_files = files[i]
					continue
				for fname in files[i]:
					self.get_training_file(fname.split()[0],fname.split()[1])
				self.get_wc()

			for fname in test_files:
				true_class = fname.split()[1]
				pred_class = self.test_file(fname.split()[0])
				if true_class == pred_class:
					correct += 1
				total += 1
			f_acc = correct/total
			acc += f_acc

		print (acc/k)
		return acc/k
