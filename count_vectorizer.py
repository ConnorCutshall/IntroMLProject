import numpy as np

""" Helper Functions """
def get_count_vectorizer(text, vocab):
	# Set up the dict_vector
	dict_vector = {}
	for word in vocab:
		dict_vector[word] = 0

	# Populate the dict_vector
	for word in text:
		dict_vector[word] += 1
	
	# Add the values to vector
	vector = []
	for word in dict_vector:
		vector.append(dict_vector[word])

	return vector

""" Main Function """
def get_KNN_vectors(author_texts, vocab):

	author_vectors = {}
	for author in author_texts:
		author_vectors[author] = []
		texts = author_texts[author]

		# Step 2: Convert each document to a count vector
		for text in texts:
			author_vectors[author].append(get_count_vectorizer(text, vocab))

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
