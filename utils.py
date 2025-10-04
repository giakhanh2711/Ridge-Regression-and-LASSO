import re
import pandas as pd

def parse_txt_file_to_csv(txt_filename, csv_filename):
    with open(txt_filename) as f:
        raw_data = f.read()
    formatted_data = re.sub(r"\S+:", ",", raw_data)

    with open(csv_filename) as f:
        f.write(formatted_data)


def load_data_to_array(csv_filename):
    df = pd.read_csv(csv_filename)
    data = df.to_numpy() # First Column for Y, the rest for X
    X = data[:, 1:]
    Y = data[:, 0]
    return X, Y