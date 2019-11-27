# Settings
from dotenv import load_dotenv
import os
load_dotenv()
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
TRANSACTIONS_FOLDER = os.getenv("TRANSACTIONS_FOLDER")
epochs = os.getenv("epochs")
num_layers = os.getenv("num_layers")
batch_size = os.getenv("batch_size")
data = os.getenv("data")
list_products = os.getenv("list_products")
prediction_length = os.getenv("prediction_length")
freq = os.getenv("freq")
min_date = os.getenv("min_date")
max_date = os.getenv("max_date")



epochs=35
num_layers=5
batch_size=50

data="LAN.csv"

list_products=["CONF TONIQUE B400ML","GENIFIQUE 13 SERUM B75ML","ADV GEN EYES LIGHT PEARLB20ML"]

prediction_length=60
freq="D"
min_date="2016-07-01"
max_date="2019-06-30"

TRANSACTIONS_FOLDER="P:/0. R&D/6. SKU sales forecast/1_Raw data/1_LAN_AAD_data"
OUTPUT_FOLDER="P:/0. R&D/6. SKU sales forecast/5_output"
