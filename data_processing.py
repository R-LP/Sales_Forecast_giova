import pandas as pd
import numpy as np
import os
import missingno


#-----------------------------------------------------------------------------------------
# HELPER FUNCTIONS
#-----------------------------------------------------------------------------------------

TRANSACTIONS_FOLDER = "P:/0. R&D/6. SKU sales forecast/1_Raw data/1_LAN_AAD_data"
DATA_FOLDER = "P:/0. R&D/6. SKU sales forecast/1_Raw data/2_Processed_Data"


# Return all paths to files in the transactions files folder
def get_transactions_files_list():
    files = os.listdir(TRANSACTIONS_FOLDER)
    files = [os.path.join(TRANSACTIONS_FOLDER, file) for file in files]
    return files


def read_from_processed_data_folder(filename, folder = None):
    if folder is None: folder = DATA_FOLDER
        path = os.path.join(folder, filename)
        print(f"... Read file at {path}")
        if filename.endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_pickle(path)


def is_in_processed_data_folder(filename):
    return filename in os.listdir(DATA_FOLDER)


def read_from_transactions_foler(filename):
    if filename.endswith(".csv"):
        return pd.read_csv(os.path.join(TRANSACTIONS_FOLDER, filename))
    else:
        return pd.read_pickle(os.path.join(TRANSACTIONS_FOLDER, filename))


def save_to_processed_data_folder(filename):
    path = os.path.join(DATA_FOLDER, filename)
    data.to_csv(path, index = False)
    print(f"... File saved at {}".format(path))


def save_pickle_to_processed_data_folder(data, name):
    path = os.path.join(DATA_FOLDER, name)
    data.to_pickle(path)
    print(f"... File saved at {}".format(path))    


def get_period_list(min_date = "2015-01-01", max_date = "2018-12-31"):
    return pd.date_range(min_date, max_date,  freq = 'W')


def convert_to_week_date(x, periods):
    if pd.isnull(x):
        return x
    else:
        if x < periods[0] or x > periods[-1]:
            return None
        else:
            return periods[periods.get_loc(x, method = "pad")]


def save_json(d, filepath):
    with open(filepath, "w", encoding ="utf8") as file:
        json.dump(d, file, indent = 4, sort_keys = True)


def read_json(filepath):
    return json.loads(open(filepath, "r", encoding = "utf8").read())


