import numpy as np
import math

def NN(train_features, train_labels, test_features, ord=2): # order of the norm to pass to the linalg.norm function
	"""
	Nearest neighbor algorithm where
	train_features contains training data (size ntrain x nfeatures),
	train_labels   contains training labels (size ntrain), and
	test_features  contains test data (size ntest x nfeatures).

	Returns:
	- test_kNN: an ntest-length vector of the estimated class for each test point
	- min_index: an ntest-length vector of the training data point that is closest
				to each test data point
	"""
	ntest = test_features.shape[0]
	test_kNN = np.zeros(ntest, dtype=train_labels.dtype)
	min_index = np.zeros(ntest, dtype=int)

	for i in range(ntest):
		distances = np.linalg.norm(train_features - test_features[i], axis=1, ord=ord)
		min_index[i] = np.argmin(distances)
		test_kNN[i] = train_labels[min_index[i]]

	return test_kNN, min_index

def calc_accuracy(true_labels, est_labels):
	"""
	Calculates the per-class and overall accuracy of the predictions.

	Parameters:
	- true_labels: vector of true class labels, must be nonnegative integers but not necessarily sequential
	- est_labels: vector of estimated class labels.

	Returns:
	- class_accuracies: vector of class accuracies. The i'th element is the accuracy of class i
	- overall_accuracy: the overall accuracy
	"""
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

def update_centroids(features, cluster_ids, k):
	"""
	Completes the step of k means where you update the centroids
	based on the current estimate of the clusters.

	inputs:
	- features:    ndata x nfeatures matrix of data
	- cluster_ids: ndata array of current cluster IDs (assumed to be ints)
	- k:           number of clusters

	returns:
	- centroids:   k x nfeatures matrix where each row is the updated centroid,
					calculated as the average of all data points assigned to that
					cluster ID
	"""

	ndata, nfeatures = features.shape    # define dimension variables
	centroids = np.zeros((k, nfeatures)) # initialize centroids with zeros

	# Compute the new centroids as the mean of points assigned to each cluster
	for i in range(k):
		cluster_points = features[cluster_ids == i]  # Extract points in cluster i
		if len(cluster_points) > 0:
			centroids[i] = np.mean(cluster_points, axis=0)  # Compute mean for each feature

	return centroids