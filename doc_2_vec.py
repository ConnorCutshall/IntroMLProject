import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

""" Helper Functions """
def train_doc2vec_model(texts, vector_size=50, epochs=40):
	"""Train a Doc2Vec model on given list of texts"""
	tagged_data = [TaggedDocument(words=text.lower().split(), tags=[str(i)])
				   for i, text in enumerate(texts)]

	model = Doc2Vec(vector_size=vector_size, min_count=1, epochs=epochs)
	model.build_vocab(tagged_data)
	model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

	return model

""" Main Function """
def get_KNN_vectors(author_texts, vector_size=50):
	author_vectors = {}

	for author in author_texts:
		texts = author_texts[author]
		model = train_doc2vec_model(texts, vector_size=vector_size)

		author_vectors[author] = []
		for text in texts:
			words = text.lower().split()
			vec = model.infer_vector(words)
			author_vectors[author].append(np.array(vec))

	return author_vectors

""" Test """
def test():
	print("doc2vec TEST:")

	author_texts = {
		"donny": [
			"the cat sat on the mat",
			"the dog sat on the log",
			"cats and dogs are friends"
		]
	}

	author_vectors = get_KNN_vectors(author_texts, vector_size=10)
	for author in author_vectors:
		print(author + ":")
		for vec in author_vectors[author]:
			print(vec)

# Uncomment to run:
test()
