import numpy as np

""" Helper Functions """
def compute_vocab(texts):
	"""Extract sorted list of unique words across all documents"""
	vocab_set = set()
	for doc in texts:
		for word in doc.lower().split():
			vocab_set.add(word)
	return sorted(list(vocab_set))

def compute_count_vector(text, vocab):
	"""Compute raw count vector for a single document"""
	words = text.lower().split()
	counts = {word: 0 for word in vocab}

	for word in words:
		if word in counts:
			counts[word] += 1

	vector = []
	for word in vocab:
		vector.append(counts[word])
	
	return np.array(vector)

""" Main Function """
def get_KNN_vectors(author_texts):
	author_vectors = {}

	for author in author_texts:
		author_vectors[author] = []
		texts = author_texts[author]

		# Step 1: Learn vocabulary from all texts by this author
		vocab = compute_vocab(texts)

		# Step 2: Convert each document to a count vector
		for text in texts:
			author_vectors[author].append(compute_count_vector(text, vocab))

	return author_vectors

""" Test """
def test():
	print("count_vectorizer TEST:")

	author_texts = {
		"donny": [
			"the cat sat on the mat",
			"the dog sat on the log",
			"cats and dogs are friends"
		]
	}

	author_vectors = get_KNN_vectors(author_texts)
	for author in author_vectors:
		print(author + ":")
		for vec in author_vectors[author]:
			print(vec)

# Uncomment to run:
#test()
