from settings import *
import pandas as pd
import numpy as np
import os
import missingno
from gluonts.dataset.common import ListDataset
import matplotlib.pyplot as plt
from datetime import datetime
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    Chain,
    FieldName,
    #ExpectedNumInstanceSampler,
    #FieldName,
    #InstanceSplitter,
    #SetFieldIfNotPresent,
)


# The class data inherits from the class of the object intance
class Data(object):

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
    def save_to_processed_data_folder(self, filename):
        path = os.path.join(DATA_FOLDER, filename)
        self.data.to_csv(path, index = False)
        print("File saved at {}".format(path)) 


    def show_missing_values(self):
        missingno.matrix(self.data)
        plt.show()


    def convert_column_to_datetime(self, col):
        self.data[col] = pd.to_datetime(self.data[col])

    
    @staticmethod
    def get_period_list(min_date = "2016-07-01", max_date = "2019-06-30", freq = 'W'):
        if freq is not 'W':
            daterange = pd.date_range(min_date, max_date,  freq = freq, tz = None).to_pydatetime()
            daterange = pd.DataFrame(daterange)
            daterange.columns = ["OrderDate"]
            return daterange
        else:
            daterange = pd.date_range(min_date, max_date,  freq = 'W', tz = None).to_pydatetime()
            daterange = pd.DataFrame(daterange)
            daterange.columns = ["OrderDate"]
            return daterange


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


class TransactionsData(Data):
    def __init__(self, filename):
        print(f">> Loading Transaction Data")
        self.data = self.read_from_transactions_folder(filename)
        self.data = self.data[["OrderDate","ProductEnglishname", "SalesQuantity", "SalesAmount"]]
        self.data["SalesAmount"] = self.data["SalesAmount"].map(float)
        self.data["SalesQuantity"] = self.data["SalesQuantity"].map(float)
        self.to_datetime(force = True)
        self.create_temporal_features("OrderDate")


    # Groupby ProductEnglishName with different granularities and return a column with correct date format
    def groupby_product(self, granularity, min_date, max_date):
        if (granularity == "day") or (granularity == "D"):
            print(f">> Aggregating transactions at productenglishname level and daily granularity")
            self.data = self.data.groupby(["ProductEnglishname", "year", "month", "day"],
                                            as_index = False)["SalesQuantity"].sum()
            self.period_list = self.get_period_list(min_date, max_date, freq = 'D')
            self.data["OrderDate"] = pd.to_datetime(self.data[["year", "month", "day"]])

        if (granularity == "week") or (granularity == "W"):
            print(f">> Aggregating transactions at productenglishname level and weekly granularity")
            self.data = self.data \
                .groupby("ProductEnglishname", as_index = False) \
                .apply(lambda x:x.set_index("OrderDate").resample("W").agg({"ProductEnglishname" : "first", "SalesQuantity" : "sum"})) \
                .reset_index() \
                .drop("level_0", axis = 1)
            self.data["OrderDate"] = pd.to_datetime(self.data[["year", "month", "day"]])
            self.period_list = self.get_period_list(min_date, max_date, freq = 'W')

        if (granularity == "month") or (granularity == "M"):
            print(f">> Aggregating transactions at productenglishname level and monthly granularity")
            self.data = self.data \
                .groupby("ProductEnglishname", as_index = False) \
                .apply(lambda x:x.set_index("OrderDate").resample("M").agg({"ProductEnglishname" : "first", "SalesQuantity" : "sum"})) \
                .reset_index() \
                .drop("level_0", axis = 1)
            self.data["OrderDate"] = pd.to_datetime(self.data[["year", "month", "day"]])       
            self.period_list = self.get_period_list(min_date, max_date, freq = 'M')

        self.data["ProductEnglishname"] = self.data["ProductEnglishname"].fillna(method = "ffill")
        self.data["SalesQuantity"][self.data["SalesQuantity"] < 0] = 0
        return self.data


    # ProductEnglishname is a list of product
    def Product_sales(self, list_products, granularity, min_date, max_date):
        if type(list_products) is not list:
            print(f">> The product entered as an input should be a list of product")
        self.groupby_product(granularity, min_date, max_date)
        self.data = self.data.loc[self.data["ProductEnglishname"].isin(list_products)]
        self.data = self.data.groupby(["OrderDate"], as_index = False)["SalesQuantity"].sum()
        if (granularity == "day") or (granularity == "D"):
            self.data = self.period_list.merge(self.data, how = 'left', on = "OrderDate")
        self.data["SalesQuantity"].fillna(0, inplace = True)
        self.create_temporal_features("OrderDate")
        self.data["OrderDate"] = pd.to_datetime(self.data[["year", "month", "day"]])
        return self.data


    def train_test_set(self, list_products, prediction_length, freq, min_date, max_date):
        self.data_final = self.Product_sales(list_products = list_products, granularity = freq, min_date = min_date,  max_date = max_date)
        # Creating different fields for the training dataset
        [f"FieldName.{k} = '{v}'" for k, v in FieldName.__dict__.items() if not k.startswith('_')]
        train = self.data_final[:-prediction_length]
        # Ability to add feat_static_cat (static categorical), feat_static_real (static real), feat_dynamic_cat, feat_dynamic_real (dyamic)
        List_features = list(self.data_final.columns)
        List_features.remove("OrderDate")
        List_features.remove("SalesQuantity")
        if freq == "M":
            List_features.remove("Day")
        min_date = pd.to_datetime(min_date, yearfirst= True)
        max_date = pd.to_datetime(max_date, yearfirst = True)

        # Drawing test and train sets
        self.train_ds = ListDataset([{FieldName.TARGET: train.SalesQuantity, 
                                FieldName.START: pd.Timestamp(min_date, freq = freq, unit = freq),
                                FieldName.FEAT_DYNAMIC_REAL: train[List_features],
                                }], 
                                freq=freq)

        self.test_ds = ListDataset([{FieldName.TARGET: self.data_final.SalesQuantity, 
                        FieldName.START: pd.Timestamp(min_date, freq = freq, unit = freq),
                        FieldName.FEAT_DYNAMIC_REAL: self.data_final[List_features],
                        }], 
                        freq=freq)

        return self.train_ds, self.test_ds