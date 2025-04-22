import os
import FileReader
import tf_idf
import classification
import random
import numpy as np

# Data Gathering
wd = os.getcwd()  # set working directory here if not using project folder
data_path = os.path.join(wd, "Data")

def split_data(data_dict, train_ratio= 0.8):
    train_dict = {}
    test_dict = {}

    for author, texts in data_dict.items():
        random.shuffle(texts)  # Shuffle the texts for randomness
        split_index = 2  # int(len(texts) * train_ratio)

        train_dict[author] = texts[:split_index]  # Training data
        test_dict[author] = texts[split_index:]  # Testing data

    return train_dict, test_dict

# Data Vectorization
def do_clean_texts(texts):
    clean_texts = []
    for text in texts:
        clean_texts.append(text.lower().split())
    return clean_texts

def get_vocab(train_dict, test_dict):

    vocab = []
    # For Each Texr
    for dict in [train_dict, test_dict]:
        for author in dict.keys():
            for text in dict[author]:

                # Add unique words
                for word in text:
                    if word not in vocab:
                        vocab.append(word)
    vocab.sort()
    return vocab

def author_vectors_to_features_and_labels(author_vectors):
    # author_vectors --> {author1: [vec1, vec2...], author2: [vec3, vec4...]}

    # Basic
    train_features = []
    train_labels = []

    # For Each Label
    for label in author_vectors.keys():
        for vector in author_vectors[label]:
            train_features.append(vector)
            train_labels.append(label)
    
    # Convert to NP Arrays
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    return train_features, train_labels

# Main
def main():

    ## Get the Raw Data
    data_dict = FileReader.read_text_files_by_author(data_path)
    train_dict, test_dict = split_data(data_dict)
    authors = np.array(data_dict.keys())

    print("Training data size:", {author: len(texts) for author, texts in train_dict.items()})
    print("Testing data size:", {author: len(texts) for author, texts in test_dict.items()})

    # Clean the texts to they are of the same format
    for author in train_dict.keys():
        train_dict[author] = do_clean_texts(train_dict[author])
    for author in test_dict.keys():
        test_dict[author] = do_clean_texts(test_dict[author])

    ## Get the Author Vectors
    vocab = get_vocab(train_dict, test_dict)

    # TODO: Make a check that tests out different algorithms
    if True:
        author_vectors_train = tf_idf.get_KNN_vectors(train_dict, vocab)
        author_vectors_test = tf_idf.get_KNN_vectors(test_dict, vocab)

    ## Get the True Labels for the Test

    ## Get the KNN Vectorized Data
    train_features, train_labels = author_vectors_to_features_and_labels(author_vectors_train)
    test_features, test_true_labels = author_vectors_to_features_and_labels(author_vectors_test)

    print("-------------")
    print("Train Features shape: " + str(train_features.shape))
    print("Train Features: " + str(train_features))
    print("Train labels shape: " + str(train_labels.shape))
    print("Train Labels: " + str(train_labels))
    print("-------------")
    print("Test Features shape: " + str(test_features.shape))
    print("Test Features: " + str(test_features))
    print("Test true_Labels shape: " + str(test_true_labels.shape))
    print("Test true_Labels: " + str(test_true_labels))
    print("-------------")

    ## NN Classification
    test_est_labels, _min_index = classification.NN(train_features, train_labels, test_features)

    print("-------------")
    print("Test est_Labels shape: " + str(test_est_labels.shape))
    print("Test est_Labels: " + str(test_est_labels))
    print("-------------")


if __name__ == "__main__":
    main()

