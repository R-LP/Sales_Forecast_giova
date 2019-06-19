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

 