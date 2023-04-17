"""
This module provides various utility functions for handling data input and
output operations, such as saving and loading different file formats.
"""

import contextlib
import gzip
import json
import pickle
import random
import string
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy
import pandas
from StyleFrame import StyleFrame, Styler, utils


def save(data: list, path_to_file: str, file_name: str = None) -> None:
    """
    Save a list of data to a file.

    :param data: The data to be saved (must be a list).
    :param path_to_file: The path to the directory, the file should be saved.
    :param file_name: An optional argument to specify the file name (default:
                      a unique combination of timestamp and random characters).
    :raises AssertionError: If the data type is not a list.
    """
    if not isinstance(data, list):
        raise AssertionError('Data type must be list!')

    if file_name is None:
        timestamp = int(time.time())
        random_string = ''.join(random.choices(string.ascii_lowercase, k=4))
        file_name = f'{timestamp}_{random_string}'

    file = f'{path_to_file}{file_name}.txt'

    with open(file, 'w+') as f:
        for item in data:
            f.write(f'{item}\n')


def save_excel(filename: str, line_number: int, data: dict) -> None:
    """
    Modifies and saves an Excel file with the given filename, updating
    the specified line with new data.

    Args:
        filename: The name of the Excel file to update.
        line_number: The line number to update in the Excel file.
        data: A dictionary containing the data to update in the Excel file.
    """
    # Modify values in dataframe.
    df = pandas.read_excel(f"../data/excel_files/{filename}.xlsx")
    df.loc[line_number, "eer(dev)"] = min(data["dev_eer"])
    df.loc[line_number, "eer(eval)"] = data["eval_eer"]

 # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = StyleFrame.ExcelWriter(f"../data/EXCELS/{filename}.xlsx")

    # Convert the dataframe to an XlsxWriter Excel object.
    sf = StyleFrame(df, Styler(bold=False, font_size=15, font_color="blue"))
    Styler(bg_color="blue", bold=False, font=utils.fonts.arial, font_size=15)
    sf.set_column_width(columns=["eer(dev)", "eer(eval)"], width=15)
    sf.set_column_width(columns=["dense_units", "activation"], width=40)
    rows = [str(row) for row in range(1, len(df))]
    sf.set_row_height(rows=rows, height=30)
    sf.to_excel(writer, sheet_name="mlp_parameters")

    # Overwrite data to excel file.
    writer.save()


def save_figure(text: str, filename: str, data: list,
                path_to_figure: str) -> None:
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


def save_pickle(path_to_save: str, data_to_dump: object):
    """
    Saves data as a gzipped pickle file.

    Args:
        path_to_save: The file path to save the gzipped pickle file.
        data_to_dump: The data to be pickled and saved.

    Returns:
        True if the file was saved successfully.
    """
    with gzip.open(path_to_save, "wb") as f:
        pickle.dump(data_to_dump, f)

    return True


def save_h5_file(path_to_write: str, data_to_write: str,
                 dataset_name: str = "data") -> bool:
    """
    Saves data to an .h5 file.

    Args:
        path_to_write: The file path where the .h5 file will be created.
        data_to_write: The data to be written to the .h5 file.
        dataset_name: The name of the dataset in the .h5 file.

    Returns:
        True if the file was saved successfully.
    """
    with contextlib.suppress(FileExistsError):
        with h5py.File(f"{path_to_write}.h5", "w") as file_handler:
            file_handler.create_dataset(dataset_name, data_to_write)
    return True


def load_line(file_path: str, line_number: int) -> str:
    """
    Returns a specific line from the file.

    Args:
        file_path: The target file path.
        line_number: The target line number.

    Returns:
        The specified line from the file.
    """
    with open(file_path) as f:
        all_lines = f.readlines()
    return all_lines[line_number]


def load_pickle(path_to_file: str) -> object:
    """
    Loads data from a gzipped pickle file.

    Args:
        path_to_file: The file path to load the gzipped pickle file.

    Returns:
        The loaded data from the pickle file.
    """
    with gzip.open(path_to_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return data


def load_json(path_to_file: str) -> object:
    """
    Loads data from a JSON file.

    Args:
        path_to_file: The file path to the JSON file.

    Returns:
        The loaded data from the JSON file.
    """
    with open(path_to_file, "r") as file_handler:
        json_file = json.load(file_handler)

    return json_file


def load_text(path_to_file: str) -> str:
    """
    Load data from a file using NumPy's genfromtxt function.

    Args:
        path_to_file: The path to the file that should be loaded.

    Returns:
        A NumPy array containing the data from the file.
    """
    return numpy.genfromtxt(path_to_file, delimiter=' ', dtype=str)


def load_excel(filename: str) -> pandas.DataFrame:
    """
    Loads an Excel file and returns its contents as a DataFrame.

    Args:
        filename: The name of the Excel file to load.

    Returns:
        A pandas DataFrame containing the contents of the Excel file.
    """
    return pandas.read_excel(f"../data/excel_files/{filename}")


def load_h5_file(path_to_read: str) -> object:
    """
    Loads data from an .h5 file.

    Args:
        path_to_read: The file path to the .h5 file.

    Returns:
        The data from the .h5 file.
    """
    try:
        with h5py.File(f"{path_to_read}.h5", "r") as hf:
            data = []
            data = hf[data][:]
    except FileExistsError as e:
        print(e)

    return data


def matrix_to_vector(matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Converts a given matrix (m, n) into a vector ((m * n), 1).

    Args:
        matrix: A numpy array with a shape (m, n).

    Returns:
        A numpy array of shape (m * n,).
    """
    return matrix.flatten()


def flush_process_info(message: str, index: int, no_of_files: int) -> None:
    """
    Prints the progress information on the console.

    Args:
        message: String to display on the console.
        index: An integer representing the current index to calculate progress.
        no_of_files: An integer representing the total number of files to
                     calculate progress percentage.
    """
    # Clear the current line on the console.
    sys.stdout.write("\r")

    # Calculate progress percentage, create the info message, and display it on
    # the console.
    progress_percentage = 100 * ((index + 1) / no_of_files)
    sys.stdout.write(f"%%%d{message}" % progress_percentage)
    sys.stdout.flush()


def combine_data(data_list: list, delimiter: str = "") -> list:
    """
    Combines the given data into a single list.

    Args:
        data_list: A list of data sets to combine together.
        delimiter: Delimiter to place between data set elements.

    Returns:
        A list containing combined elements from the given data sets.
    """
    data_list = [data for data in data_list if data is not None]

    if not data_list:
        return []

    lengths = [len(data) for data in data_list]
    assert all(length == lengths[0] for length in lengths), "Check lengths!"

    return [delimiter.join(str(data[i]) for data in data_list)
            for i in range(lengths[0])]
