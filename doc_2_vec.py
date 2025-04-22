import numpy as np

""" Helper Functions """
def build_vocab(texts):
	"""Collect all unique words from a list of documents"""
	vocab = set()
	for text in texts:
		for word in text.lower().split():
			vocab.add(word)
	return sorted(list(vocab))

def initialize_embeddings(vocab, dim=50):
	"""Assign a random embedding vector to each word"""
	embeddings = {}
	for word in vocab:
		embeddings[word] = np.random.randn(dim)
	return embeddings

def document_embedding_with_window(text, embeddings, dim=50, window_size=2):
	"""Compute document vector using average of word vectors in context window"""
	words = text.lower().split()
	vec = np.zeros(dim)
	count = 0

	for i in range(len(words)):
		# Window from (i - window_size) to (i + window_size)
		start = max(0, i - window_size)
		end = min(len(words), i + window_size + 1)

		for j in range(start, end):
			word = words[j]
			if word in embeddings:
				vec += embeddings[word]
				count += 1

	if count > 0:
		vec /= count

	return vec

""" Main Function """
def get_KNN_vectors(author_texts, dim=50, window_size=2):
	author_vectors = {}

	for author in author_texts:
		texts = author_texts[author]
		vocab = build_vocab(texts)
		embeddings = initialize_embeddings(vocab, dim)

		author_vectors[author] = []
		for text in texts:
			vec = document_embedding_with_window(text, embeddings, dim, window_size)
			author_vectors[author].append(vec)

	return author_vectors

""" Test """
def test():
	print("Simple Doc2Vec w/ window TEST:")

	author_texts = {
		"donny": [
			"the cat sat on the mat",
			"the dog sat on the log",
			"cats and dogs are friends"
		]
	}

	author_vectors = get_KNN_vectors(author_texts, dim=10, window_size=1)
	for author in author_vectors:
		print(author + ":")
		for vec in author_vectors[author]:
			print(vec)

# Uncomment to run:
# test()
