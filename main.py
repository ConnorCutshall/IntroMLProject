import os
import FileReader
import tf_idf
import classification
import random
import numpy as np

################################################## Constants
## NN Classification
# Order
ORDER_START = 0
ORDER_END = 2
TOTAL_ORDERS = 2
# Lamda
LAMDAA_START = 0
LAMDAA_END = 10
LAMDAA_ORDERS = 2
##################################################


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

    ## Overwrite Test (comment out of not using)
    """
    train_dict = {
        "John": ["A black cat sat on a mat", "A blue dog likes to eat the mat"],
        "Barry": ["The flash the flash", "I am barry allen, the flash", "zoom"]
        }
    test_dict = {"John": ["The cat and dog are a mat that flash flash like barry allen, he goes zoom I am"]}
    """

    # Clean the texts to they are of the same format (turn texts into a formatted array of words)
    print("-------------")
    for author in train_dict.keys():
        train_dict[author] = do_clean_texts(train_dict[author])
        print("Training (" + author + "): " + str(len(train_dict[author])))
    for author in test_dict.keys():
        test_dict[author] = do_clean_texts(test_dict[author])
        print("Testing (" + author + "): " + str(len(test_dict[author])))
    print("-------------")

    ## Get the Author Vectors
    vocab = get_vocab(train_dict, test_dict)

    # TODO: Make a check that tests out different algorithms
    author_vectors_train = {}
    author_vectors_test = {}
    if True:

        # Make the IDF Dict
        author_vectors_train = tf_idf.get_KNN_vectors(train_dict, vocab)

        # Push in the test data bit at a time (don't want to train it on itself!)
        author_vectors_test = {}
        for author in test_dict.keys():
            author_vectors_test[author] = []
            for text in test_dict[author]:
                author_vector = tf_idf.get_KNN_vectors({author: [text]}, vocab, True)
                author_vectors_test[author].append(author_vector[author][0])

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
    print("-------------")
    print("Test Features shape: " + str(test_features.shape))
    print("Test Features: " + str(test_features))
    print("Test true_Labels shape: " + str(test_true_labels.shape))
    print("Test true_Labels: " + str(test_true_labels))
    print("-------------")

    ## NN Classification
    # For each Order
    print("CSV Output: ")
    csv_out = "Order, Lambda, Accuracy,"
    for order_idx in range(TOTAL_ORDERS + 1):
        order = ORDER_START + (ORDER_END - ORDER_START)*(float(order_idx)/TOTAL_ORDERS)

        # For each Lamdaa
        for lambdaa_idx in range(LAMDAA_ORDERS + 1):
            lambdaa = LAMDAA_START + (LAMDAA_END - LAMDAA_START)*(float(lambdaa_idx)/LAMDAA_ORDERS)

            # Get the Chosen Label
            test_est_labels, _min_index = classification.NN(train_features, train_labels, test_features, order, lambdaa)
            _class_accuracies, overall_accuracy = classification.calc_accuracy(test_true_labels, test_est_labels)

            # Add to CSV Out
            csv_out += "\n" + str(order) + ", " + str(lambdaa) + ", " + str(overall_accuracy) + ", "
    
    print(csv_out)
            


if __name__ == "__main__":
    main()

