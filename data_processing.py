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


# The class data inherits from the class of the object intance
class Data(object):

    @staticmethod
    def is_in_processed_data_folder(filename):
        return filename in os.listdir(DATA_FOLDER)

    # Return all paths to files in the transactions files folder
    @staticmethod
    def get_transactions_files_list():
        files = os.listdir(TRANSACTIONS_FOLDER)
        files = [os.path.join(TRANSACTIONS_FOLDER, file) for file in files]
        return files


    @staticmethod
    def read_from_processed_data_folder(filename, folder = None):
        if folder is None: 
            folder = DATA_FOLDER
        path = os.path.join(folder, filename)
        print(f"... Read file at {path}")
        if filename.endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_pickle(path)  

            
    @staticmethod
    def read_from_transactions_folder(filename):
        if filename.endswith(".csv"):
            return pd.read_csv(os.path.join(TRANSACTIONS_FOLDER, filename))
        else:
            return pd.read_pickle(os.path.join(TRANSACTIONS_FOLDER, filename))


    @staticmethod
    def save_to_processed_data_folder(filename):
        path = os.path.join(DATA_FOLDER, filename)
        data.to_csv(path, index = False)
        print("File saved at {}".format(path)) 


    def show_missing_values(self):
        missingno.matrix(self.data)
        plt.show()


    def convert_column_to_datetime(self, col):
        self.data[col] = pd.to_datetime(self.data[col])

    
    def convert_column_to_weekdate(self, col):
        print(f">> converting column {col} to weekdate")
        periods = get_period_list()
        self.data[col] = self.data[col].map(lambda x: convert_to_week_date(x, periods))


    @staticmethod
    def get_period_list(min_date = "2016-04-01", max_date = "2018-12-31", freq = 'W'):
        if freq is not 'W':
            return pd.date_range(min_date, max_date,  freq = freq)
        else:
            return pd.date_range(min_date, max_date,  freq = 'W')


    @staticmethod
    def convert_to_week_date(x, periods):
        if pd.isnull(x):
            return x
        else:
            if x < periods[0] or x > periods[-1]:
                return None
            else:
                return periods[periods.get_loc(x, method = "pad")]


    # To fix with L'Oreal format (pd.to_datetime)
    def to_datetime(self, force = False):
        cols = ["OrderDate"]
        for col in cols:
            if col in self.data.columns:
                if force or not hasattr(self, "converted_date_cols") or col not in self.converted_date_cols:
                    print(f">> converting column {col} to datetime")
                    self.convert_column_to_datetime(col)
                    if hasattr(self, "converted_date_cols"):
                        self.converted_date_cols.append(col)
                    else:
                        self.converted_date_cols = [col]
            else:
                print("Not in columns")


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
            keys =list(keys.data["ProductEnglishname"])
        self.data = self.data.loc[self.data["ProductEnglishname"].isin(keys)]


# Class to use
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
        print(">> Aggregating transacions at product and store level")
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


# DATA MANIPULATION
def clean_returns(self):
    self.data.loc[self.data["SalesQuantity"] < 0, "SalesQuantity"] = 0.0


def compute_weekly_temporal_features(self):
    print(">> Computing weekly temporal features")
    self.data["week_of_month"] = self.data["OrderDate"].map(lambda x: f"WN{(x.day//7)+1}")
    self.data["week_of_year"] = self.data["OrderDate"].map(lambda x : f"W{x.week}")


class TransactionsProduct(TransactionsAggregated):
    def __init__(self, data = None):
        super().__init__(data = data)

    # -----------------------------------------------------------------------------------
    # BUILD DATA

    def get_data_from_file(self, filepath, keys = None, weekly = True, inplace = True):
        data = TransactionsMonthlyGranular(filepath)
        if keys is not None:
            data.filter_on_products(keys)
        data = data.groupby_product(weekly = weekly)
        if inplace:
            self.data = data
        else:
            return data


    def build_time_series_data(self):
        print(">> Pivoting to time series data")
        data = self.data.set_index(["OrderDate", "ProductEnglishname"]).unstack("ProductEnglishname")
        data = data.reindex(get_period_list())
        data = data.fillna(0.0)
        data.columns = data.columns.get_level_values(1)
        return TransactionsProductTS(data)


def TransactionsProductTS(Data):
    def __init__(self, data = None, filename = None):
        super().__init__(data = data)
        if self.data is None:
            self.data = read_from_processed_data_folder(filename)

        def show_time_series(self):
            pass

# function to test on different kind of products (split on products)
def train_test_split(self, test_size = 0.3, shuffle = False):
    print(">> Spllitting in train test set")
    unique_products = self.data["ProductEnglishname"].unique()
    if shuffle:
        random.shuffle(unique_products)
    cut = int(len(unique_products)*(1-test_size))
    products_train = unique_products[:cut]
    products_test = unique_products[cut:]
    print(f"... Among the {len(unique_products)} : {len(products_train)} in training set and {len(products_test)} in test set")

    # Actually splitting the data
    train = self.data.loc[self.data["ProductEnglishname"].isin(products_train)]
    test = self.data.loc[self.data["ProductEnglishname"].isin(products_test)]

    # Retrieving X and y
    target = "SalesQuantity"
    X_train, y_train = train.drop(target, axis = 1), train[target]
    X_test, y_test = test.drop(target, axis = 1), test[target]

    return X_train, y_train, X_test, y_test


class TransactionsProductStore(TransactionsAggregated):
    def __init__(self, data = None):
        super().__init__(data = data)
        self.name = "transactions_at_product_store_level"

    def get_data_from_file(self, filepath, keys = None, weekly = True, inplace = True):
        data = TransactionsMonthlyGranular(filepath)
        if keys is not None:
            data.filter_on_products(keys)
        data = data.groupby_product_store(weekly = weekly)
        if inplace:
            self.data = data
        else:
            return data

    def get_data_from_files(self, filepaths, keys = None, weekly = True, inplace = True):
        if isinstance(filepaths, str): filepaths = [filepaths]
        for i, filepath in enumerate(tqdm_notebook(filepaths)):
            df = self.get_data_from_file(filepath, keys, weekly, inplace = False)
            if i == 0:
                data = df
            else:
                data = data.append(df, ignore_index = True)

        if weekly:
            data = data.groupby(["OrderDate", "ProductEnglishname", "CounterLocalName"], as_index = False).sum()
        if inplace:
            self.data = data
        else:
            return data

        




    






























