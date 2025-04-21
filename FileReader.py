import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def read_text_files(directory):
    text_data = []
    file_names = []

    # Use glob to find all .txt files in the directory
    for file_path in glob.glob(os.path.join(directory, '*.txt')):
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data.append(file.read())
            file_names.append(os.path.basename(file_path))  # Store the file name for reference

    return text_data, file_names


# Directory containing your .txt files
directory = 'path/to/your/text/files'

# Read the text files
documents, file_names = read_text_files(directory)

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert the TF-IDF matrix to a DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=file_names)

# Display the TF-IDF DataFrame
print(tfidf_df)

