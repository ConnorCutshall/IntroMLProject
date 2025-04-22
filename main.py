import os
import FileReader
import re

import tf_idf
import count_vectorizer
import doc_2_vec

import classification

import random
import numpy as np

################################################## Constants
## NN Classification

# Order
ORDER_START = 0
ORDER_END = 2
TOTAL_ORDERS = 20
# K
K_START = 1
K_END = 4
K_ORDERS = 3
# Dev Testing
CHOSEN_ALGO = "tf_idf" # Choosen from ["tf_idf", "count_vectorizer", "doc_2_vec"]
IGNORE_CSV = False
OMIT_FEATURES_FROM_PRINTOUT = True
##################################################


# Data Gathering
wd = os.getcwd()  # set working directory here if not using project folder
data_path = os.path.join(wd, "Data")

def split_data(data_dict, train_ratio= 0.8):
    train_dict = {}
    test_dict = {}

    for author, texts in data_dict.items():
        random.shuffle(texts)  # Shuffle the texts for randomness
        split_index = int(len(texts) * train_ratio)

        train_dict[author] = texts[:split_index]  # Training data
        test_dict[author] = texts[split_index:]  # Testing data

    return train_dict, test_dict

# Data Vectorization
def do_clean_texts(texts):
    clean_texts = []
    for text in texts:
        clean_texts.append([])
        for word in text.lower().split():
            clean_texts[-1].append(remove_punctuation(word))
    return clean_texts

def remove_punctuation(text):
    return ''.join(char for char in text if char.isalpha() or char == ' ')


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

# KNN Algo
def calc_accuracy(train_features, train_labels, test_features, test_true_labels, k, order, lambdaa):
    test_est_labels, _min_index = classification.KNN(train_features, train_labels, test_features, k, order, lambdaa)
    _class_accuracies, overall_accuracy = classification.calc_accuracy(test_true_labels, test_est_labels)
    return overall_accuracy


# File Saving
def save_string_to_file(content, filename):
    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    # Write the content to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"Saved to: {file_path}")


# Main
def main():

    ## Get the Raw Data
    data_dict = FileReader.read_text_files_by_author(data_path)
    raw_train_dict, raw_test_dict = split_data(data_dict)
    train_dict = {}
    test_dict = {}

    # Clean the texts to they are of the same format (turn texts into a formatted array of words)
    print("-------------")
    for author in raw_train_dict.keys():
        train_dict[author] = do_clean_texts(raw_train_dict[author])
        print("Training (" + author + "): " + str(len(train_dict[author])))
    for author in raw_test_dict.keys():
        test_dict[author] = do_clean_texts(raw_test_dict[author])
        print("Testing (" + author + "): " + str(len(test_dict[author])))
    print("-------------")

    ## Get the Vocab
    vocab = get_vocab(train_dict, test_dict)

    ## Get the Author Vectors
    vector_size = 0
    vector_count = 0
    author_vectors_train = {}
    author_vectors_test = {}

    if CHOSEN_ALGO == "tf_idf":

        vector_size = len(vocab)

        author_vectors_train = tf_idf.get_KNN_vectors(train_dict, vocab)
        author_vectors_test = tf_idf.get_KNN_vectors(test_dict, vocab)
        """
        author_vectors_test = {}
        for author in test_dict.keys():
            author_vectors_test[author] = []
            for text in test_dict[author]:
                author_vector = tf_idf.get_KNN_vectors({author: [text]}, vocab, True)
                author_vectors_test[author].append(author_vector[author][0])
        """
    
    elif CHOSEN_ALGO == "count_vectorizer":
        vector_size = len(vocab)

        author_vectors_train = count_vectorizer.get_KNN_vectors(train_dict, vocab)
        author_vectors_test = count_vectorizer.get_KNN_vectors(test_dict, vocab)

    elif CHOSEN_ALGO == "doc_2_vec":
        # Initialization Variables
        DIMENSION = 50
        WINDOW_SIZE = 2

        # Actual Implementaton
        vector_size = DIMENSION

        embeddings = doc_2_vec.initialize_embeddings(vocab, DIMENSION)
        author_vectors_train = doc_2_vec.get_KNN_vectors(train_dict, embeddings, DIMENSION, WINDOW_SIZE)
        author_vectors_test = doc_2_vec.get_KNN_vectors(test_dict, embeddings, DIMENSION, WINDOW_SIZE)

    else:
        return

    ## Get the True Labels for the Test

    ## Get the KNN Vectorized Data
    train_features, train_labels = author_vectors_to_features_and_labels(author_vectors_train)
    print("-------------")
    print("Train Features shape: " + str(train_features.shape))
    if not OMIT_FEATURES_FROM_PRINTOUT:
        print("Train Features: " + str(train_features))
    print("Train labels shape: " + str(train_labels.shape))
    print("Train Labels: " + str(train_labels))
    print("-------------")

    test_features, test_true_labels = author_vectors_to_features_and_labels(author_vectors_test)
    print("-------------")
    print("Test Features shape: " + str(test_features.shape))
    if not OMIT_FEATURES_FROM_PRINTOUT:
        print("Test Features: " + str(test_features))
    print("Test true_Labels shape: " + str(test_true_labels.shape))
    print("Test true_Labels: " + str(test_true_labels))
    print("-------------")

    ## Temporary Print

    ## Misc Data
    vector_count = 0
    magnitude = 0
    for author in author_vectors_train.keys():
        for vector in author_vectors_train[author]:
            vector_count += 1

            mag_squared = 0
            for x in vector:
                mag_squared += pow(x, 2)
            magnitude += pow(mag_squared, 0.5)
    
    magnitude /= vector_count

    word_count = 0
    for author in train_dict.keys():
        for text in train_dict[author]:
            word_count += len(text)
    
    word_count /= vector_count

    ## Simple Print
    print("-------------")
    print("Vector Dimension: " + str(vector_size))
    print("Avg Vector Magnitude: " + str(magnitude))
    print("Avg Document Word Count: " + str(word_count))
    print("-------------")

    ## NN Classification
    csv_out = "Order, K, Accuracy,"
    for order_idx in range(TOTAL_ORDERS + 1):
        order = ORDER_START + (ORDER_END - ORDER_START)*(float(order_idx)/TOTAL_ORDERS)

        # For each K
        for K_idx in range(K_ORDERS + 1):
            K = int(K_START + (K_END - K_START)*(float(K_idx)/K_ORDERS))

            accuracy = calc_accuracy(train_features, train_labels, test_features, test_true_labels, K, order, 0)

            csv_out += "\n" + str(order) + ", " + str(K) + ", " + str(accuracy) + ", "

    if not IGNORE_CSV:
        save_string_to_file(csv_out, CHOSEN_ALGO + ".csv")
            

if __name__ == "__main__":
    main()

