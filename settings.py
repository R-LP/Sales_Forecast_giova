# Settings
from dotenv import load_dotenv
import os
load_dotenv()
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
TRANSACTIONS_FOLDER = os.getenv("TRANSACTIONS_FOLDER")
list_products=os.getenv("list_products")


# data file
data="LAN.csv"


# Setting the name of the algorithm - choices ar: Arima, DeepAR or Prophet
algorithm = "DeepAR"

# DeepAR hyperparameters
epochs=35
num_layers=5
batch_size=50

# Prophet hyperparameters
mcmc_samples=300
changepoint_prior_scale=0.01
interval_width=0.9
seasonality_mode='multiplicative'
weekly_seasonality=True
daily_seasonality=True
regressor_list = []


# prediction frame
prediction_length=60
freq="D"
min_date="2016-07-01"
max_date="2019-06-30"

TRANSACTIONS_FOLDER="P:/0. R&D/6. SKU sales forecast/1_Raw data/1_LAN_AAD_data"
OUTPUT_FOLDER="P:/0. R&D/6. SKU sales forecast/5_output"
list_list_products = [["CONF TONIQUE B400ML","GENIFIQUE 13 SERUM B75ML"],["ADV GEN EYES LIGHT PEARLB20ML"]]