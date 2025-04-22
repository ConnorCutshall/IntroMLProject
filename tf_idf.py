import numpy as np
import math

""" Helper Functions """
def compute_tf(text, vocab):
	"""Compute term frequency vector for a document"""

	# Get the Frequency Dictionary
	freq_dict = {}
	for word in vocab:
		freq_dict[word] = 0

	# Populate the Freq_Dict
	for word in text:
		freq_dict[word] += 1
	
	# Get the Average of each Word's appearence
	tf_dict = {}
	for word in freq_dict:
		tf_dict[word] = freq_dict[word] / len(vocab)

	return tf_dict

def compute_idf(texts, vocab):
	"""Compute IDF values for all terms in the corpus"""
	# Count how many documents contain each word

	# For Each Word
	idf_dict = {}
	N = len(texts)
	for word in vocab:

		# Get the amount of times a word in all of the documents
		doc_count = 0
		for text in texts:
			if word in texts:
				doc_count += 1
		idf_dict[word] = N / (1 + doc_count)#math.log(N / (1 + doc_count))  # Add 1 to avoid division by zero

	return idf_dict

def compute_tfidf_vector(tf_dict, idf_dict):
	"""Compute TF-IDF vector using a precomputed IDF dictionary"""

	# Construct the TF-IDF vector
	tfidf_vector = []
	for word in idf_dict:
		tfidf_vector.append(tf_dict[word] * idf_dict[word])

	return tfidf_vector

""" Main Function """
def get_KNN_vectors(author_texts, vocab):

	# For Each Author
	author_vectors = {}
	for author in author_texts:
		author_vectors[author] = []

		# Get the Texts and the idf_dict
		texts = author_texts[author]
		idf_dict = compute_idf(texts, vocab)

		# For Each Text, Compute the Author Vector
		for text in texts:
			tf_dict = compute_tf(text, vocab)
			author_vectors[author].append(compute_tfidf_vector(tf_dict, idf_dict))

	return author_vectors

""" Test """
def test():
	print("tf_idf TEST:")

	author_texts = {
		"donny": [
			"the cat sat on the mat",
			"the dog sat on the log",
			"cats and dogs are friends"
		]
    }
	author_vectors = get_KNN_vectors(author_texts)
	for author in author_vectors:
		print(author + ": " + str(author_vectors[author]))

