## gluonts.__version__ = '0.4.2'
## fbprophet.__version__ = '0.5' Embedded inside GluonTs
## pyramid.__version__ = '1.10.4'

# Settings
from dotenv import load_dotenv
import os
import json
load_dotenv(os.path.join(os.getcwd(), "forecast_env.env"))
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
TRANSACTIONS_FOLDER = os.getenv("TRANSACTIONS_FOLDER")
PROMO_DATA_FOLDER = os.getenv("PROMO_DATA_FOLDER")
list_products=json.loads(os.getenv("list_products"))


# Data file
data="data_test.csv"
# Mapping inputs
input_cols_mapping = {'OrderDate_test':'OrderDate',
                      'Productname':'ProductEnglishname',
                      'SalesQuantity_test':'SalesQuantity'}

# Promo data file
#promo_data="Promo_data.csv"
promo_data=None

# Setting the name of the algorithm - choices ar: ARIMA, DeepAR, Prophet or All
algorithm = "ARIMA"

# DeepAR hyperparameters
epochs=5
num_layers=2
batch_size=32

# Prophet hyperparameters
mcmc_samples=10 #300
changepoint_prior_scale=0.01
interval_width=0.9
seasonality_mode='multiplicative'
weekly_seasonality=True
daily_seasonality=True
regressor_list = []
prophet_params = {'mcmc_samples' : mcmc_samples, 'changepoint_prior_scale' : changepoint_prior_scale, 'interval_width' : interval_width, 'seasonality_mode' : seasonality_mode,
                        'weekly_seasonality' : weekly_seasonality, 'daily_seasonality' : daily_seasonality}



# Prediction frame settings
prediction_length=10
freq="D"
min_date="2016-01-01"
max_date="2019-10-31"