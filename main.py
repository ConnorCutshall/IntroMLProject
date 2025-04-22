import os
import FileReader


wd = os.getcwd()  # set working directory here if not using project folder
data_path = os.path.join(wd, "Data")


def main():
    data_dict = FileReader.read_text_files_by_author(data_path)
    print(data_dict)


if __name__ == "__main__":
    main()

