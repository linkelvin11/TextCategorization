## Overview

This project was completed in partial requirement for ECE467 Natural Language Processing at The Cooper Union. The goal of this project is to implement a text categorization system.

## Build Instructions

```
git clone https://github.com/linkelvin11/TextCategorization.git;
cd TextCategorization
```
### Minimum Requirements

python 3.4+

## Table of Contents
<!-- MarkdownTOC depth=0 -->

- [Text Categorization][#text-categorization]
	- [Smoothing][#smoothing]
- [Tokenization][#tokenization]
	- [Stop Words][#stop-words]
- [Other Datasets][#other-datasets]
	- [K-Folding][#k-folding]

<!-- /MarkdownTOC -->

## Text Categorization [#text-categorization]

Text categorization for this project was done using a naive bayes classifier. The naive bayes classifer assumes that each feature is independent from the next. To create the classifier word and document counts needed to be generated for the following categories:

- Number of documents per class (Nc)
- Number of documents total (N)
- Number of tokens per class (tc)
- Number of each token in each class (tt)

Using these word/document count values the probability that a particular document belongs to any given class can be calculated based on its tokens.

Using the formula `cmap = argmax[log(Nc) + SUM[log(tc/tt)]]` the most likely class can be determined.

### Smoothing [#smoothing]

The word counts for each token were smoothed using laplace (add-one) smoothing. This ensures that any tokens that appear in the test set but not the training set will not be zero. Since the log probability is used to determine which class any given text belongs to, a token with probability zero will cause the log probability to be -inf.

## Tokenization [#tokenization]

The nltk regexp tokenizer was used for this project. The regexp tokenizer is built upon the nltk punkt tokenizer, which filters out punctuation. The regexp tokenizer allows for specification of a regex to filter out any unwanted tokens. The regex used during tokenization was `'[a-zA-z\.]+'` which will allow only words and acronyms.

### Stop Words [#stop-words]

A list of stop words was also used in order to increase the accuracy of the classifier. The stop list contains commonly used english words (such as 'a', 'the', 'I') which could appear in any document. Using a stop list allows the tokenizer to skip commonly used words, meaning that the classification will be based only on words more specific to a given topic.

When used with corpus 1, the stop list only showed marginal gains in accuracy.

## Other Datasets [#other-datasets]

### K-Folding [#k-folding]

To test corpus 2 and corpus 3, k-fold cross validation was used. k-folding tests the training set against itself using a holdout set. This is done by training the classifier using (k-1)/k of the original dataset, and validating the trained classifier using the remaining 1/k of the dataset. 

A k value of 10 was used for corpora 2 and 3, which resulted in an accuracy of 0.864 and 0.788 respectively


