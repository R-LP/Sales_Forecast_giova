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
    #ExpectedNumInstanceSampler,
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
    def read_from_transactions_folder(filename):
        if filename.endswith(".csv"):
            return pd.read_csv(os.path.join(TRANSACTIONS_FOLDER, filename))
        else:
            return pd.read_pickle(os.path.join(TRANSACTIONS_FOLDER, filename))

    @staticmethod
    def read_from_promo_folder(filename_promo):
        if (filename_promo is not None):
            if filename_promo.endswith(".csv"):
                return pd.read_csv(os.path.join(PROMO_DATA_FOLDER, filename_promo))
            else:
                return pd.read_pickle(os.path.join(PROMO_DATA_FOLDER, filename_promo))


    def show_missing_values(self):
        missingno.matrix(self.data)
        plt.show()


    def convert_column_to_datetime(self, col):
        self.data[col] = pd.to_datetime(self.data[col])

    
    @staticmethod
    def get_period_list(min_date = min_date, max_date = max_date, freq = freq):
        daterange = pd.date_range(min_date, max_date,  freq = freq, tz = None)
        daterange = pd.DataFrame(daterange)
        daterange.columns = ["OrderDate"]
        daterange['OrderDate'] = pd.to_datetime(daterange['OrderDate'], format='%Y-%m-%d', errors='ignore')
        return daterange


    def create_temporal_features(self, df, column):
        date_column = pd.to_datetime(df[column], format='%Y-%m-%d', errors='ignore')
        df["year"] = date_column.map(lambda x : x.year)
        df["month"] = date_column.map(lambda x : x.month)
        df["week"] = date_column.map(lambda x : x.week)
        df["day"] = date_column.map(lambda x : x.day)
        df["dayofweek"] = date_column.map(lambda x : x.dayofweek)
        return(df)


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
    def __init__(self, filename, filename_promo):
        print(f">> Loading Transaction Data")
        self.data = self.read_from_transactions_folder(filename)
        self.data = self.data[["OrderDate","ProductEnglishname", "SalesQuantity", "SalesAmount"]]
        self.data["SalesAmount"] = self.data["SalesAmount"].map(float)
        self.data["SalesQuantity"] = self.data["SalesQuantity"].map(float)
        self.promo_data = self.read_from_promo_folder(filename_promo)


    # ProductEnglishname is a list of product
    def Product_sales(self, list_products, granularity, min_date, max_date):
        self.period_list = self.get_period_list(min_date, max_date, freq = granularity)
        self.data["SalesQuantity"][self.data["SalesQuantity"] < 0] = 0
        self.data["OrderDate"] = pd.to_datetime(self.data['OrderDate'], format='%Y-%m-%d', errors='ignore')
        if type(list_products) is not list:
            print(f">> The product entered as an input should be a list of product")
        print(f">> Aggregating transactions at productenglishname level and at granularity %s" %granularity)
        for i, product in enumerate(list_products):
            _col_name = "list_" + str((i+1))
            _data = self.data.loc[self.data["ProductEnglishname"].isin([product])]
            _data = _data.rename(columns={'SalesQuantity' : _col_name}, inplace = False)
            _data.set_index(pd.DatetimeIndex(_data['OrderDate']), inplace=True)
            _data = _data.drop(columns=['OrderDate'])
            _data = _data.drop(columns=['SalesAmount'])
            _data = _data.resample(granularity).sum().reset_index()
            _data['OrderDate'] = pd.to_datetime(_data['OrderDate'], format='%Y-%m-%d', errors='ignore')
            if (i == 0):
                agg_data = self.period_list.merge(_data, how = 'left', on = "OrderDate")
            else:
                agg_data = agg_data.merge(_data, how = 'left', on = "OrderDate")
            agg_data[_col_name].fillna(0, inplace = True)

        agg_data = self.create_temporal_features(agg_data, "OrderDate")
        #print(f">>agg_data tail")
        #print(agg_data.tail(5))
        return agg_data


    def agg_promo_data(self, granularity, min_date, max_date):
        self.period_list = self.get_period_list(min_date, max_date, freq = granularity)
        self.promo_data['OrderDate'] = pd.to_datetime(self.promo_data['OrderDate'], format='%Y-%m-%d', errors='ignore')
        self.promo_data.set_index(pd.DatetimeIndex(self.promo_data['OrderDate']), inplace=True)
        self.promo_data = self.promo_data.drop(columns=['OrderDate'])
        self.promo_data = self.promo_data.resample(granularity).sum().reset_index()
        self.promo_data['OrderDate'] = pd.to_datetime(self.promo_data['OrderDate'], format='%Y-%m-%d', errors='ignore')
        agg_promo_data = self.period_list.merge(self.promo_data, how = 'left', on = "OrderDate")
        #print(f">>agg_promo_data  tail")
        #print(agg_promo_data.tail(5))
        return agg_promo_data


    def train_test_set(self, list_products, prediction_length, freq, min_date, max_date):
        self.data_final = self.Product_sales(list_products = list_products, granularity = freq, min_date = min_date,  max_date = max_date)
        if self.promo_data is not None:
            self.data_final = self.data_final.merge(self.agg_promo_data(granularity=freq, min_date=min_date, max_date = max_date), how = 'left', on = "OrderDate")
        # Creating different fields for the training dataset
        train = self.data_final[:-prediction_length]
        # In List_features we add the columns with feat_static_cat (static categorical), feat_static_real (static real), feat_dynamic_cat, feat_dynamic_real (dyamic)
        self.List_features = list(self.data_final.columns)
        self.List_features.remove("OrderDate")
        self.List_product_no = ["list_"+str(i+1) for i in range(len(list_products))]

        self.List_features = list(set(self.List_features)-set(['list_1', 'list_2', 'list_3']))

        if ((freq == "M") or (freq == "Month") or (freq == "W") or freq == "Week"):
            self.List_features.remove("day")
        min_date = pd.to_datetime(min_date, yearfirst= True)
        max_date = pd.to_datetime(max_date, yearfirst = True)
        
        
        # Drawing test and train sets
        if algorithm=='DeepAR':
            self.train_ds = ListDataset([{'target': train[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq),
                                          'feat_dynamic_real': train[self.List_features]}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.train_ds.list_data[p]['target'] = self.train_ds.list_data[p]['target'][0]

            self.test_ds = ListDataset([{'target': self.data_final[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq),
                                          'feat_dynamic_real': self.data_final[self.List_features]}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.test_ds.list_data[p]['target'] = self.test_ds.list_data[p]['target'][0]

        elif algorithm=='Prophet':
            self.train_ds = ListDataset([{'target': train[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq)}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.train_ds.list_data[p]['target'] = self.train_ds.list_data[p]['target'][0]

            self.test_ds = ListDataset([{'target': self.data_final[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq)}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.test_ds.list_data[p]['target'] = self.test_ds.list_data[p]['target'][0]
        return self.train_ds, self.test_ds


    def train_future_set(self, list_products, prediction_length, freq, min_date, max_date, add_promo_data=False):
        future_date = (pd.to_datetime(max_date) + pd.Timedelta(value=prediction_length, unit=freq)).strftime('%Y-%m-%d')
        self.data_final_future = self.Product_sales(list_products = list_products, granularity = freq, min_date = min_date,  max_date = max_date)
        if self.promo_data is not None:
            self.data_final_future = self.data_final_future.merge(self.agg_promo_data(granularity=freq, min_date=min_date, max_date = max_date), how = 'left', on = "OrderDate")
        self.period_list_future = self.get_period_list(min_date, future_date, freq = freq)
        self.data_final_future = self.period_list_future.merge(self.data_final_future, how = 'left', on = "OrderDate")
        self.data_final_future = self.create_temporal_features(self.data_final_future, "OrderDate")
        # Creating different fields for the training dataset
        train_total = self.data_final_future[:-prediction_length]
        # In List_features we add the columns with feat_static_cat (static categorical), feat_static_real (static real), feat_dynamic_cat, feat_dynamic_real (dyamic)
        self.List_features = list(self.data_final_future.columns)
        self.List_features.remove("OrderDate")
        self.List_product_no = ["list_"+str(i+1) for i in range(len(list_products))]

        self.List_features = list(set(self.List_features)-set(['list_1', 'list_2', 'list_3']))

        if ((freq == "M") or (freq == "Month") or (freq == "W") or freq == "Week"):
            self.List_features.remove("day")
        min_date = pd.to_datetime(min_date, yearfirst= True)
        max_date = pd.to_datetime(max_date, yearfirst = True)
        
        
        # Drawing test and train sets
        if algorithm=='DeepAR':
            self.train_final_ds = ListDataset([{'target': train_total[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq),
                                          'feat_dynamic_real': train_total[self.List_features]}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.train_final_ds.list_data[p]['target'] = self.train_final_ds.list_data[p]['target'][0]

            self.future_ds = ListDataset([{'target': self.data_final_future[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq),
                                          'feat_dynamic_real': self.data_final_future[self.List_features]}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.future_ds.list_data[p]['target'] = self.future_ds.list_data[p]['target'][0]
        
        elif algorithm=='Prophet':
            self.train_final_ds = ListDataset([{'target': train_total[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq)}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.train_final_ds.list_data[p]['target'] = self.train_final_ds.list_data[p]['target'][0]

            self.future_ds = ListDataset([{'target': self.data_final_future[list_product].values, 
                                          'start': pd.Timestamp(min_date, freq=freq)}
                                         for list_product in self.List_product_no],
                                        freq=freq,
                                        one_dim_target = False)
            for p in range(len(self.List_product_no)):
                self.future_ds.list_data[p]['target'] = self.future_ds.list_data[p]['target'][0]
        return self.train_final_ds, self.future_ds

