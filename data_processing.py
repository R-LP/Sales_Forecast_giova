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


    @staticmethod
    def get_period_list(min_date = "2019-01-01", max_date = "2019-03-31", freq = 'W'):
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


    def convert_column_to_weekdate(self, col, freq = 'W'):
        print(f">> converting column {col} to weekdate")
        periods = self.get_period_list(freq = freq)
        self.data[col] = self.data[col].map(lambda x: self.convert_to_week_date(x, periods))


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
        self.data = self.read_from_processed_data_folder(name, folder)


    def to_csv(self, name = None):
        if name is None: name = self.name
        self.data = self.save_to_processed_data_folder(name, name)


    def get_shape(self):
        return self.data.shape

    
    def get_len(self):
        return len(self.data)


    def get_columns(self):
        return self.data.columns
    

    # keys input is a list
    def filter_on_products(self, keys):
        print(f">> Filtering on given list of products")
        if not isinstance(keys, list):
            keys =list(keys.data["ProductEnglishname"])
        self.data = self.data.loc[self.data["ProductEnglishname"].isin(keys)]


# Class to use
class TransactionsMonthlyGranular(Data):
    def __init__(self, filename):
        print(f">> Loading Monthly Transaction Data")
        self.data = self.read_from_transactions_folder(filename)
        #self.data = pd.read_csv(filepath)
        self.data = self.data[["CustomerID", "OrderID", "OrderDate","SalesQuantity",
        "SalesAmount", "Axe", "ProductCategory", "ProductSubCategory",
        "ProductLine", "ProductSubLine", "ProductEnglishname", "CounterLocalName"]]
        self.data["SalesAmount"] = self.data["SalesAmount"].map(float)
        self.data["SalesQuantity"] = self.data["SalesQuantity"].map(float)
        self.to_datetime(force = True)
        self.create_temporal_features("OrderDate")


    # Groupby ProductEnglishName with different granularities and return a column with correct date format
    def groupby_product(self, granularity):
        if granularity == "day":
            print(f">> Aggregating transactions at productenglishname level and daily granularity")
            self.data = self.data.groupby(["ProductEnglishname", "year", "month", "day"],
                        as_index = False)["SalesQuantity"].sum()

        if granularity == "week":
            print(f">> Aggregating transactions at productenglishname level and weekly granularity")
            self.data = self.data.groupby(["ProductEnglishname", "year", "month", "week"],
                        as_index = False)["SalesQuantity"].sum()

        if granularity == "month":
            print(f">> Aggregating transactions at productenglishname level and monthly granularity")
            self.data = self.data.groupby(["ProductEnglishname", "year", "month"],
                        as_index = False)["SalesQuantity"].sum()

        self.data["ProductEnglishname"] = self.data["ProductEnglishname"].fillna(method = "ffill")
        self.data["SalesQuantity"][self.data["SalesQuantity"] < 0] = 0
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



       




    






























