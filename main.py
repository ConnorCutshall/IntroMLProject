import os


def read_text_files_in_directory(directory):
    text_contents = {}

    # List all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a text file
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_contents[filename] = file.read()

    return text_contents
