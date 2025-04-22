import numpy as np
import math

def NN(train_features, train_labels, test_features, order=2, lambaa=0): # order of the norm to pass to the linalg.norm function
	ntest = test_features.shape[0]
	test_labels = np.zeros(ntest, dtype=train_labels.dtype)
	min_index = np.zeros(ntest, dtype=int)

	for i in range(ntest):
		
		# Get the distances
		base = get_distance(train_features - test_features[i], order)
		penalty = lambaa*(np.count_nonzero(test_features))
		distances = base + penalty

		# Get the best fitting label
		min_index[i] = np.argmin(distances)
		test_labels[i] = train_labels[min_index[i]]

	return test_labels, min_index

def KNN(train_features, train_labels, test_features, K=3, order=2, lambaa=0):
	ntest = test_features.shape[0]
	test_labels = np.empty(ntest, dtype=train_labels.dtype)
	neighbors_index = np.zeros((ntest, K), dtype=int)

	for i in range(ntest):
		# Compute distances with optional sparsity penalty
		base = get_distance(train_features - test_features[i], order)
		penalty = lambaa * np.count_nonzero(test_features[i])
		distances = base + penalty

		# Find indices of K smallest distances
		k_indices = np.argsort(distances)[:K]
		neighbors_index[i] = k_indices

		# Majority vote using simple counting
		k_labels = train_labels[k_indices]
		label_counts = {}
		for label in k_labels:
			label_counts[label] = label_counts.get(label, 0) + 1
		test_labels[i] = max(label_counts, key=label_counts.get)

	return test_labels, neighbors_index


def get_distance(vector, order):
	return np.linalg.norm(vector, axis=1, ord=order)

def calc_accuracy(true_labels, est_labels):
	# Compute overall accuracy
	overall_accuracy = np.mean(true_labels == est_labels)

	# Identify unique classes
	unique_classes = np.unique(true_labels)
	class_accuracies = np.zeros(len(unique_classes))

	# Compute accuracy for each class
	for i, cls in enumerate(unique_classes):
		class_mask = true_labels == cls
		class_accuracies[i] = np.mean(est_labels[class_mask] == cls)

	return class_accuracies, overall_accuracy
