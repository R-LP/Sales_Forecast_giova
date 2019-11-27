# Settings
from dotenv import load_dotenv
import os
load_dotenv()
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
TRANSACTIONS_FOLDER = os.getenv("TRANSACTIONS_FOLDER")
list_products=os.getenv("list_products")

# Neural network hyperparameters
epochs=35
num_layers=5
batch_size=50

# data file
data="LAN.csv"

# prediction frame
prediction_length=60
freq="D"
min_date="2016-07-01"
max_date="2019-06-30"

TRANSACTIONS_FOLDER="P:/0. R&D/6. SKU sales forecast/1_Raw data/1_LAN_AAD_data"
OUTPUT_FOLDER="P:/0. R&D/6. SKU sales forecast/5_output"
