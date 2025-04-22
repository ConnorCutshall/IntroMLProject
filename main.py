import os
import FileReader
import tf_idf
import classification
import random
import numpy as np



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


def main():

    data_dict = FileReader.read_text_files_by_author(data_path)
    train_dict, test_dict = split_data(data_dict)

    print("Training data size:", {author: len(texts) for author, texts in train_dict.items()})
    print("Testing data size:", {author: len(texts) for author, texts in test_dict.items()})

    authors = np.array(data_dict.keys())
    train_labels = np.array(train_dict.keys())

    train_features = np.array(tf_idf.get_KNN_vectors(train_dict).values())
    test_features = np.array(tf_idf.get_KNN_vectors(test_dict).values())
    print("Train features shape:", train_features.shape)
    print("Test features shape:", test_features.shape)

    # train_features = np.array(tf_idf.get_KNN_vectors(train_dict).values())
    # test_features = np.array(tf_idf.get_KNN_vectors(test_dict).values())

    print(test_features)

    classification.NN(train_features, train_labels, test_features)


if __name__ == "__main__":
    main()

