# -*- coding: utf-8 -*-
import gzip
import json
import os
import pickle
import sys
from configparser import ConfigParser, ExtendedInterpolation

import h5py
import keras
import matplotlib.pyplot as plt
import numpy
import pandas
from StyleFrame import StyleFrame, Styler, utils

from features import ASVspoof


def matrix_to_vector(matrix):
    """
    Splits given matrix (m,n) into vector ((mxn),1).

    Args:
        matrix: Matrix shaped input data.

    Returns:
        Data vector.
    """
    # Find the shape of matrix and vector size to be created.
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]
    v_size = row_size * col_size
    vector = numpy.ones([v_size], dtype=float)
    for index in range(len(matrix)):
        a = index * col_size
        b = (index + 1) * col_size
        vector[a:b] = matrix[index, :]

    return vector


def pickle_data(path_to_save, data_to_dump):
    with gzip.open(path_to_save, "wb") as f:
        pickle.dump(data_to_dump, f)

    return True


def unpickle_data(path_to_file):
    with gzip.open(path_to_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return data


def load_json(path_to_file):
    with open(path_to_file, "r") as file_handler:
        json_file = json.load(file_handler)

    return json_file


def load_text(path_to_file):
    return numpy.genfromtxt(path_to_file, delimiter=" ", dtype=str)


def read_config(param="config"):
    """
    Loads config file.

    Args:
        param: Param

    Returns:
        Dictionary
    """
    working_dir = os.getcwd()
    os.chdir("../config")

    with open("file_structure.json", "r") as fp:
        project_structure = fp.read()

    project_structure = json.loads(project_structure)

    os.chdir(working_dir)

    return project_structure[param]


def flush_process_info(message, index, no_of_files):
    """
    Prints out the process info.

    Args:
        message: String to flush on console.
        index: An integer to calculate process.
        no_of_files: An integer to calculate process percentage.
    """
    # Delete current line on console.
    sys.stdout.write("\r")

    # Calculate process percentage, create info message and flush to console.
    count = 100 * ((index + 1) / no_of_files)
    sys.stdout.write(("%%%d" + message) % count)
    sys.stdout.flush()


def save_h5_file(path_to_write, data_to_write, dataset_name="data"):
    """
    Save h5 file.

    Args:
        path_to_write: Full path of file to be created.
        data_to_write: Data to be written.
        dataset_name: Name of the dataset.

    Returns:
        Boolean Indicator
    """
    try:
        with h5py.File(path_to_write + ".h5", "w") as file_handler:
            file_handler.create_dataset(dataset_name, data_to_write)
    except FileExistsError:
        pass

    return True


def read_h5_file(path_to_read):
    """
    Load `.h5` formatted files.

    Args:
        path_to_read: path of the file.

    Returns:
        Data
    """
    try:
        with h5py.File(path_to_read + ".h5", "r") as hf:
            data = []
            data = hf[data][:]
    except FileExistsError as e:
        print(e)

    return data


def change_working_dir(param):
    """
    Changes working directory.

    Args:
        param: Folder to be root directory.
    """
    looping = True
    while looping:
        list_dir = os.listdir(os.getcwd())
        if param in list_dir:
            os.chdir(param + "/")
            looping = False
        elif len(os.getcwd()) > 8:
            os.chdir("../")
        else:
            assertion_text = "\nSet directory to {} folder.".format(param)
            raise AssertionError(assertion_text)


def clear_dir(path, extension):
    """
    Removes files in directory.

    Args:
        path: Path to directory.
        extension: File extensions in path.
    """
    # List files in directory.
    files = os.listdir(path)

    # Remove files in directory.
    print("Deleting files...")

    for file in files:
        if file.endswith(extension):
            os.remove(path + "/" + file)
        else:
            pass
    print("{} files deleted in {}".format(extension, path))


def save_fig(text, filename, data, path_to_figure):
    """
    Writes figure into disk as jpeg.

    Args:
        text: Text to write into figure.
        filename: Figure filename.
        data: Data to create figure.
        path_to_figure: Path to file.
    """
    plt.plot(data)
    plt.xlabel("Iteration", fontsize=15, color="blue")
    plt.ylabel("Equal Error Rate", fontsize=15, color="red")
    plt.text(0, 50, text, fontsize=14, color="red")
    plt.savefig(path_to_figure + filename)
    plt.close()


def delete_files(path, extension):
    """
    A function to delete files with given extensions in given directory.

    Args:
        path: Required file path.
        extension: File types in path.
    """
    os.chdir(path)
    files = [f for f in os.listdir(".") if f.endswith(extension)]
    for f in files:
        os.remove(f)
    print("Files with {} extension deleted in {}".format(extension, path))


def combine_data(data_1=None, data_2=None, data_3=None, delimiter=""):
    """
    A function to combine given data into one list.

    Args:
        data_1: First data set to combine.
        data_2: Second data set to combine.
        data_3: Third data set to combine.
        delimiter: Delimiter to place between data set elements.

    Returns:
        A list gathered from given data sets.
    """
    assertion_message = "Parameters must have same length to combine together."

    if len(data_1) == 0:
        assert len(data_2) == len(data_3), assertion_message
    elif len(data_2) == 0:
        assert len(data_1) == len(data_3), assertion_message
    elif len(data_3) == 0:
        assert len(data_1) == len(data_2), assertion_message
    else:
        assert len(data_1) == len(data_2) and len(data_2) == len(
            data_3), assertion_message

    final_data = []
    for i in range(0, len(data_1)):
        final_data.append(str(data_1[i]) + delimiter + str(data_2[i]) +
                          delimiter + str(data_3[i]))

    return final_data


def save_data(file_path=None, file_name=None, data_to_save=None):
    """
    A function to save data in given path with the name of file.

    Args:
        file_path: A path in disk where file will be saved.
        file_name: A file name with desired extension which data will be
        stored.
        data_to_save: Data to save into a given file.
    """
    if file_path is not None:
        file_path = file_path + "/" + file_name
    else:
        file_path = file_name

    f = open(file_path, "w+")

    for i in range(0, len(data_to_save)):
        data_to_save[i] = data_to_save[i] + "\n"
        f.write(str(data_to_save[i]))
    f.close()


def read_line(file_path, line_number):
    """
    Returns specific line from file.

    Args:
        file_path: Target file.
        line_number: Target line.

    Returns:
        Target lines from file.
    """
    f = open(file_path)
    all_lines = f.readlines()
    result = all_lines[line_number]

    return result


def load_excel(filename):
    return pandas.read_excel("../data/excel_files/" + filename)


def read_data(dataset, subset):
    # Read data.
    path = "feats/" + dataset.upper() + "/"
    file = open(path + dataset + "_" + subset, "rb")
    data = pickle.load(file)
    # Convert to numpy array.
    x_data = numpy.asarray(data[0])
    x_data = x_data.astype("float32")
    y_data = numpy.asarray(data[1])
    # noinspection PyUnresolvedReferences
    y_data = keras.utils.to_categorical(y_data, 2)
    y_data = y_data.astype("float32")
    # Get labels and file_list.

    with open("./config/datasets.json") as fh:
        d = json.loads(fh.read())

    # noinspection DuplicatedCode
    dataset = ASVspoof(2017,
                       subset,
                       d["ASVspoof 2017"][subset]["path_to_dataset"],
                       d["ASVspoof 2017"][subset]["path_to_protocol"],
                       d["ASVspoof 2017"][subset]["path_to_wav"])

    file_list = dataset.file_list
    labels = dataset.file_list
    # Frame wise classification.
    if dataset in ["mfcc", "cqcc"]:
        indexes = data[2]
    elif dataset == "ltas":
        indexes = None
    else:
        raise AssertionError("Dataset not found.")
    if len(x_data) >= 10000:
        mini_batch_size = len(x_data) // 50
    else:
        mini_batch_size = len(x_data)
    del data

    # Store values in data dictionary.
    return {
        "dataset": dataset,
        "indexes": indexes,
        "x_data": x_data,
        "y_data": y_data,
        "labels": labels,
        "file_list": file_list,
        "mini_batch_size": mini_batch_size
    }


# noinspection DuplicatedCode
def save_excel(filename, line_number, data):
    # Modify values in dataframe.
    df = pandas.read_excel("../data/excel_files/" + filename + ".xlsx")
    df.loc[line_number, "eer(dev)"] = min(data["dev_eer"])
    df.loc[line_number, "eer(eval)"] = data["eval_eer"]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = StyleFrame.ExcelWriter("../data/EXCELS/" + filename + ".xlsx")

    # Convert the dataframe to an XlsxWriter Excel object.
    sf = StyleFrame(df, Styler(bold=False, font_size=15, font_color="blue"))
    Styler(bg_color="blue", bold=False, font=utils.fonts.arial, font_size=15)
    sf.set_column_width(columns=["eer(dev)", "eer(eval)"], width=15)
    sf.set_column_width(columns=["dense_units", "activation"], width=40)
    rows = []
    for row in range(1, len(df)):
        rows.append(str(row))
    sf.set_row_height(rows=rows, height=30)
    sf.to_excel(writer, sheet_name="mlp_parameters")

    # Overwrite data to excel file.
    writer.save()


class Config(ConfigParser):
    def __init__(self, path_to_ini_file):
        self.path_to_ini = path_to_ini_file

        # Init base class with interpolation and read ini.
        super().__init__(interpolation=ExtendedInterpolation())
        self.read(path_to_ini_file)

    def get_list(self, section, option):
        """
        Read list sections.

        Args:
            section:
            option:

        Returns:
            Section Data
        """
        # Get value and map to list.
        value = self.get(section, option)
        ret_val = list(filter(None, (x.strip() for x in value.splitlines())))

        return ret_val

    def update_values(self, section, option, value):
        """
        Updates section values.

        Args:
            section: Section
            option: Option
            value: Value

        Returns:
            Boolean Indicator
        """
        self.set(section, option, value)

        fh = open(self.path_to_ini, "w")
        self.write(fh)

        return True
