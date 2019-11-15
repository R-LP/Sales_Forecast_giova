# For neural network architecture
epochs = 35
num_layers = 5
batch_size = 50 #lower batch size means more batches need to be made, resulting in slower learning time generally
#learning_rate = 0.001 

# Parameter and Epoch Loss Logs
# 10/3/16: 6.338
# 1/3/16: 7.492
# 10/5/16: 6.169
# 10/5/8: 6.351
# 10/5/32: 6.114
# 35/5/32: 5.054
# 40/5/38: 4.929
# 50/5/38: 4.752
# 35/5/50: 


# Parameter Notes
#    Default values: epochs=100, num_layers = 2, batch_size = 32

# Create object transactions data
data = "LAN.csv"

# Products of interest
list_products = ["CONF TONIQUE B400ML",
"GENIFIQUE 13 SERUM B75ML",
"ADV GEN EYES LIGHT PEARLB20ML"
]

# Time series parameters
prediction_length = 60
freq = "D"
min_date = "2016-07-01"
max_date = "2019-06-30"

# Directories
TRANSACTIONS_FOLDER = "C:/Users/anthony.chan/Ekimetrics/Eki HK - 6. SKU sales forecast/1_Raw data/1_LAN_AAD_data"
OUTPUT_FOLDER = "C:/Users/anthony.chan/Ekimetrics/Eki HK - 6. SKU sales forecast/5_output"