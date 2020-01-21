############################################# Requirements #############################################
## gluonts.__version__ = '0.4.2'
## fbprophet.__version__ = '0.5' Embedded inside GluonTs
## pyramid.__version__ = '1.10.4'

############################################### Settings ###############################################
from dotenv import load_dotenv
import os
import json
load_dotenv(os.path.join(os.getcwd(), "forecast_env.env"))
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
TRANSACTIONS_FOLDER = os.getenv("TRANSACTIONS_FOLDER")
PROMO_DATA_FOLDER = os.getenv("PROMO_DATA_FOLDER")
list_products=json.loads(os.getenv("list_products"))


# Data file - .csv file name
data="Biotherm_HK.csv"
# Mapping inputs of you data if the following columns names are not exactly the same - replace xxx, yyy, zzz with the date, the product key and the quantity column names respectively
# input_cols_mapping = {'xxx':'OrderDate',
#                      'yyy':'ProductEnglishname',
#                      'zzz':'SalesQuantity'}
input_cols_mapping = {'Date.Transaction':'OrderDate',
                      'ProductLine':'Granulcolname',
                      'Quantity.Final':'SalesQuantity'}

# Promo data file - .csv additional data file - Make sure OrderDate is the name of the date column - If no additional data, let it be None
promo_data=None

# Setting the name of the algorithm - choices ar: ARIMA, DeepAR, Prophet or All
algorithm = "Prophet"

# Prediction frame settings
prediction_length=13 # int
freq="W" # D,W,Y
min_date="2017-06-01" # "yyyy-mm-dd"
max_date="2019-05-31" # "yyyy-mm-dd"


###################################### Algorithms hyperparameters ######################################

# DeepAR hyperparameters
epochs=5
num_layers=2
batch_size=32

# Prophet hyperparameters
mcmc_samples=0
changepoint_prior_scale=0.01
interval_width=0.9
seasonality_mode='multiplicative'
weekly_seasonality=True
daily_seasonality=True
regressor_list = []
prophet_params = {'mcmc_samples' : mcmc_samples, 'changepoint_prior_scale' : changepoint_prior_scale, 'interval_width' : interval_width, 'seasonality_mode' : seasonality_mode,
                        'weekly_seasonality' : weekly_seasonality, 'daily_seasonality' : daily_seasonality}