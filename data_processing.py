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

