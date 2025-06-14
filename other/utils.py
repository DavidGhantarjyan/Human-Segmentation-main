import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from tabulate import tabulate


RES_PREFIX = "res"
DATE_FORMAT = "%Y-%m-%d"
MODEL_NAME = "weights.pt"
EXAMPLE_FOLDER = "examples"

def find_model_in_dir_or_path(dp: str):
    if os.path.isdir(dp):
        for file in os.listdir(dp):
            if file.endswith(".pt"):
                return os.path.join(dp, file)
        raise FileNotFoundError(f"There is no model file in given directory: {dp}")
    elif os.path.isfile(dp):
        if dp.endswith(".pt"):
            return dp
        raise TypeError(f"Model file must be pytorch model: {dp}")

#
# train_results/model_name/2025-02-19_12-00-00/model_10/weights.pt
def find_last_model_in_tree(model_trains_tree_dir):
    res_dir = None  # Initialize the result directory as None.

    if os.path.exists(model_trains_tree_dir):  # Check if the directory exists.
        # `model_trains_tree_dir` -> train_results/model_name
        # `datetime.strptime(date, DATE_FORMAT)` converts a string date into a datetime object,
        # find the largest (most recent) date.
        date_objects = [datetime.strptime(date, DATE_FORMAT)
                        for date in os.listdir(model_trains_tree_dir)
                        if len(os.listdir(os.path.join(model_trains_tree_dir, date))) != 0]
        if len(date_objects) != 0:  # Check if there are valid dates.
            max_num = 0  # Initialize the maximum epoch number.
            day_dir = os.path.join(model_trains_tree_dir, max(date_objects).strftime(DATE_FORMAT))
            for name in os.listdir(day_dir):  # Iterate through the subdirectories in the latest date folder.
                st, num = name.split("_")  # Split the directory name into two parts: `st` and `num`.
                # Example: train_results/model_name/2025-02-19_12-00-00/"epoch_i"
                folder_path = os.path.join(day_dir, name)
                if max_num <= int(num) and MODEL_NAME in os.listdir(folder_path):
                    # Update `max_num` and `res_dir` if the current epoch number is larger and
                    # `MODEL_NAME` exists in the folder.
                    max_num = int(num)
                    res_dir = folder_path

    if res_dir is None:
        return None, None
    else:
        return res_dir, os.path.join(res_dir, MODEL_NAME)

def print_as_table(dataframe):
    if len(dataframe) > 4:
        print(tabulate(dataframe.iloc[[0, -3, -2, -1], :].T.fillna("---"), headers='keys'))
    else:
        print(tabulate(dataframe.T.fillna("---"), headers='keys'))

def create_new_model_trains_dir(model_trains_tree_dir):
    day_dir = os.path.join(model_trains_tree_dir, datetime.now().strftime(DATE_FORMAT))
    os.makedirs(day_dir, exist_ok=True)
    max_num = 0
    for name in os.listdir(day_dir):
        _, num = name.split("_")
        max_num = max(int(num), max_num)

    dir = os.path.join(day_dir, RES_PREFIX + "_" + str(max_num + 1))
    os.makedirs(dir, exist_ok=True)

    return dir, os.path.join(dir, MODEL_NAME)


def save_history_plot(history_table, index, title, x_label, y_label, path):
    history_dict = history_table.to_dict(orient='list')
    history_dict[index] = history_table.index.tolist()
    history_dict = {key: np.array(value) for key, value in history_dict.items()}

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    x = history_dict[index]

    for key, val in history_dict.items():
        temp_x = x
        if key == 'global_epoch':
            continue

        if np.isnan(val).any():
            non_nan_mask = ~np.isnan(val)
            temp_x = x[non_nan_mask]
            val = val[non_nan_mask]
        ax.plot(temp_x, val, label=key)
        ax.legend()

    fig.savefig(path)
    plt.close(fig)
