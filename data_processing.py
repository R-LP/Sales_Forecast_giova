import pandas as pd
import numpy as np
import os
import missingno
import matplotlib.pyplot as plt


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


# The class data inherits from the class of the object intance
class Data(object):
    def _repr_html(self):
        retunr self.data.head()._repr_html_()
    

    def show_missing_values(self):
        missingno.matrix(self.data)
        plt.show()


    def convert_column_to_datetime(self, col):
        self.data[col] = pd.to_datetime(self.data[col])

    
    def convert_column_to_weekdate(self, col):
        print(f">> converting column {col} to weekdate")
        periods = get_period_list()
        self.data[col] = self.data[col].map(lambda x: convert_to_week_date(x, periods))

    # To fix with L'Oreal format (pd.to_datetime)
    def to_datetime(self, force = False):
        cols = ["Date_Transaction"]
        for col in cols:
            if col in self.data.columns:
                if force or not hasattr(self, "converted_date_cols") or col not in self.converted_date_cols:
                    print(f">> converting column {col} to datetime")
                    self.convert_column_to_datetime(col)

                    if hasattr(self, "converted_date_cols"):
                        self.converted_date_cols.append(col)
                    else:
                        self.converted_date_cols = [col]


    def create_temporal_features(self, column):
        date_column = self.data[column]
        self.data["year"] = date_column.map(lambda x : x.year)
        self.data["month"] = date_column.map(lambda x : x.month)
        self.data["week"] = date_column.map(lambda x : x.week)
        self.data["day"] = date_column.map(lambda x : x.day)
        self.data["dayofweek"] = date_column.map(lambda x : x.dayofweek)
        self.data.drop(column, axis = 1,  inplace = True)


    def reload_from_file(self, name = None, folder = None):
        if name is None: name  = self.name
        self.data = read_from_processed_data_folder(name, folder)


    def to_csv(self, name = None):
        if name is None: name = self.name
        self.data = save_to_processed_data_folder(name, name)


    def to_pickle(self, name = None):
        if name is None: name = self.name
        save_pickle_to_processed_data_folder(self.name, name)


    def get_shape(self):
        return self.data.shape

    
    def get_len(self):
        return len(self.data)


    def get_columns(self):
        return self.data.columns


    def filter_on_products(self, keys):
        print(f">> Filtering on given list of products")

        if not isinstance(keys, list):
            keys =list(keys.data["ProductCategory"])
        self.data = self.data.loc[self.data["ProductCategory"].isin(keys)]



class TransactionsMonthlyGranular(Data):
    def __init__(self, filepath):
        print("Loading Monthly Transaction Data")
        self.data = pd.read_csv(filepath)
        self.data = self.data[["CustomerID", "OrderID", "OrderDate","SalesQuantity", "SalesAmount", "Axe", "ProductCategory", "ProductSubCategory",
        "ProductLine", "ProductSubLine", "ProductEnglishname", "CounterLocalName", "BrandName"]]
        self.data["SalesAmount"] = self.data["SalesAmount"].map(float)
        self.data["SalesQuantity"] = self.data["SalesQuantity"].map(float)

    # Groupby ProductEnglishName
    def groupby_product(self, weekly = True):
        print(">> Aggregating trnsactions at productenglishname level")
        self.data = self.data.groupby(["ProductEnglishname", "OrderDate"],  as_index = False)["SalesQuantity"].sum()
        if weekly:
            print(">> Aggregating transactions at a weekly level")
            self.to_datetime(force = True)
            self.data = self.data \
                .groupby("ProductEnglishname", as_index = False) \
                .apply(lambda x:x.set_index("OrderDate").resample("W").agg({"ProductEnglishname" : "first", "SalesQuantity" : "sum"})) \
                .reset_index() \
                .drop("level_0", axis = 1)
            self.data["ProductEnglishname"] = self.data["ProductEnglishname"].fillna(method = "ffill")
        return self.data


    def groupby_product_store(self, weekly = True):
        print(>> "Aggregating transacions at product and store level")
        self.data = self.data.groupby(["ProductEnglishname", "OrderDate", "CounterLocalName"], as_index = False)["SalesQuantity"].sum()
        if weekly:
            print(">> Aggregating transactions by week")
            self.to_datetime(force = True)
            self.data = self.data.groupby(["ProductEnglishname", "CounterLocalName"], as_index = False).apply(lambda x:x.set_index("OrderDate").resample("W"))
            self.data["ProductEnglishname"] = self.data["ProductEnglishname"].fillna(method = "ffill")
        return self.data



# Other Aggregation class
class TransactionsAggregated(Data):
    def __init__(self, data = None):
        if data is not None:
            print(">> Initialization with preloaded data")
            self.data
        else:
            print(">> Initialization with no loaded data")

    # ---------------------------------------
    # BUILD DATA
    def get_data_from_files(self, filepaths, keys = None, weekly = True, inplace = True):
        if isinstance(filepaths, str): filepaths = [filepaths]
        for i, filepath in enumerate(tqdm_notebook(filepaths)):
            df = self.get_data_from_file(filepath, keys, weekly, inplace = False)
            if i == 0:
                data = df
            else:
                data = data.append(df, ignore_index = True)

        # Groupby if weekly
        if weekly:
            data = data.groupby(["OrderDate", "ProductEnglishname"], as_index = False).sum()
            if inplace:
                self.data = data
            else:
                return data
    

    def get_data_from_transaction_folder(self, inplace = True):
        print(">> Loading data from all transactions file")
        paths = get_transactions_files_list()
        data = self.get_data_from_files(paths, inplace = False)
        if inplace:
            self.data = data
        else:
            return data


    






























