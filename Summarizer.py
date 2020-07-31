import sys
from ExtractData import ExtractData


if __name__ == "__main__":
    # check if number of arguments is valid for getting data
    if len(sys.argv) < 2:
        print("PLEASE PROVIDE FILE TO EXTRACT DATA FROM")
        exit(1)
    
    # get the name of the file that is containing the data
    file_name = list(sys.argv)[1]
    print("FILE TO EXTRACT DATA FROM: ", file_name)

    # open file and read
    with open(file_name) as f:
        text = f.read()
    
    # create a cleaning object
    extract_data = ExtractData(text)

    # extract all the sentences
    sentences = extract_data.extract_sentences()

    # calculate the tf-idf matrix
    tf_idf_matrix = extract_data.calculate_tf_idf()

