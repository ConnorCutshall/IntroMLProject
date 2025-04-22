import os
from collections import defaultdict


def read_text_files_by_author(directory):
    author_texts = defaultdict(list)  # Dictionary to hold lists of texts for each author

    # List all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a text file
        if filename.endswith('.txt'):
            author_label = os.path.splitext(filename)[0].split('_')[0]  # Get the author label from the filename
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                author_texts[author_label].append(content)  # Append content to the author's list

    return author_texts
