import os
import numpy as np
import pandas as pd


def coord_str_to_vec(coord_string):
    coord_string = str(coord_string)
    for replace_string in ["h", "m", "s", "d"]:
        coord_string = coord_string.replace(replace_string, "")
    coord_vec = coord_string.split()
    coord_vec = [float(entry) for entry in coord_vec]
    coord_vec = [np.abs(float(entry)) * np.sign(coord_vec[0]) for entry in coord_vec]

    return coord_vec


def coord_str_to_float(coord_string):
    result = np.nan
    if coord_string:
        coord_vec = coord_str_to_vec(coord_string)
        result = 0
        for i, val in enumerate(coord_vec):
            result += val / 60 ** i
    return result


def file_has_data(filename):
    if os.path.exists(filename):
        return os.stat(filename).st_size != 0
    return False


def get_file_list_with_data(file_list):
    return [filename for filename in file_list if file_has_data(filename)]


def to_numeric(df0):
    for col in df0.columns:
        if "date" not in col.lower():
            try:
                df0[col] = df0[col].apply(pd.to_numeric)
            except:
                continue
    return df0


def to_str(df0):  # to handle _HeaderCommentary entries
    for col in df0.columns:
        if col in ["COMMENT", "HISTORY"]:
            try:
                df0[col] = df0[col].apply(str)
            except:
                continue
    return df0


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def lower_cols(df):
    df.columns = [col.lower() for col in df.columns]
    return df


def flatten_cols(df):
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    return df
