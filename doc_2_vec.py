import random

""" Helper Functions """

def initialize_embeddings(vocab, dim):
	embeddings = {}
	for word in vocab:
		embeddings[word] = []
		for i in range(dim):
			embeddings[word].append(random.random())
	
	return embeddings

def document_embedding_with_window(text, embeddings, dim, window_size):
	"""Compute document vector using average of word vectors in context window"""

	# Create the Vector
	vector = []
	for i in range(dim):
		vector.append(0)

	# Populate the Dict Vector
	count = 0
	for i in range(len(text)):
		start = max(0, i - window_size)
		end = min(len(text), i + window_size + 1)

		for j in range(start, end): # Window from (i - window_size) to (i + window_size)
			word = text[j]
			count += 1

			for k in range(dim):
				vector[k] += embeddings[word][k]
	
	# Divide by the Count
	for k in range(dim):
		vector[k] /= count
	
	# Return
	return vector

""" Main Function """
def get_KNN_vectors(author_texts, embeddings, dim, window_size):
	# For Each Author
	author_vectors = {}
	for author in author_texts:
		texts = author_texts[author]

		# Process each Vector
		author_vectors[author] = []
		for text in texts:
			author_vectors[author].append(document_embedding_with_window(text, embeddings, dim, window_size))

	# Return
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
