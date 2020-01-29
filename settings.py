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
# list_products=json.loads(os.getenv("list_products"))


# Data file - .csv file name
data="GAB.csv"
# Mapping inputs of you data if the following columns names are not exactly the same - replace xxx, yyy, zzz with the date, the product key and the quantity column names respectively
# input_cols_mapping = {'xxx':'OrderDate',
#                      'yyy':'ProductEnglishname',
#                      'zzz':'SalesQuantity'}
input_cols_mapping = {'OrderDate':'OrderDate',
                      'ProductEnglishname':'Granulcolname',
                      'SalesQuantity':'SalesQuantity'}

# Promo data file - .csv additional data file - Make sure OrderDate is the name of the date column - If no additional data, let it be None
promo_data=None

# Setting the name of the algorithm - choices ar: ARIMA, DeepAR, Prophet or All
algorithm = "ARIMA"

# Prediction frame settings
prediction_length=52 # int
freq="W" # D,W,Y
min_date="2017-01-01" # "yyyy-mm-dd"
max_date="2019-12-31" # "yyyy-mm-dd"


###################################### Algorithms hyperparameters ######################################

# DeepAR hyperparameters
epochs=2
num_layers=2
batch_size=32

# Prophet hyperparameters
mcmc_samples=0
changepoint_prior_scale=0.01
interval_width=0.9
seasonality_mode='multiplicative'
yearly_seasonality=True
weekly_seasonality=True
daily_seasonality=True
regressor_list = []
prophet_params = {'mcmc_samples' : mcmc_samples, 'changepoint_prior_scale' : changepoint_prior_scale, 'interval_width' : interval_width,
                  'seasonality_mode' : seasonality_mode,
                  'weekly_seasonality' : weekly_seasonality,
                  'daily_seasonality' : daily_seasonality,
                  'yearly_seasonality' : yearly_seasonality,}

###################################### Product list to forecast ######################################
## Use "" for strings instead of '' - Can be a list of lists 
## Replace special characters by '_'

#list_products = ['LIP MAESTRO 405_MAD',  'ECSTASY LACQUER 500',  'LIP MAESTRO 400',  'LIP MAESTRO 200 CA']
list_products = ['LIP MAESTRO 405_MAD']

