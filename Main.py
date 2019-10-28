import warnings
warnings.filterwarnings("ignore")
from Predictors import *
from data_processing import *

# For prediction
prediction_length = 60
freq = "D"
min_date = "2016-07-01"
max_date = "2019-06-30"
# For neural network architecture
epochs = 10
num_layers = 3 
batch_size = 16 


# Creat the object transactions data
Transactions_obj = TransactionsData("LAN.csv")
# Create the train and test sets out of the transaction data object
train_ds, test_ds = Transactions_obj.train_test_set(list_products = list_products, prediction_length = prediction_length, min_date = min_date, max_date = max_date,
                                                        freq = freq)

# We create a Predictor object and set it as a DeeAR estimator
Predictor_instance = Predictor_sales()
Predictor_instance.define_DeepAR_predictor(freq = freq, prediction_length = prediction_length, train_ds = train_ds, epochs=epochs, num_layers = num_layers, batch_size = batch_size)
# We train the predictor object
Predictor_instance.train_predictor(train_ds = train_ds)

# Make the forecasts
forecast_it, ts_it = Predictor_instance.make_predictions(test_ds)

# Save the forecast into a csv file
for_csv, ts_csv = forecast_it, ts_it
Predictor_instance.save_csv("test", for_csv, ts_csv)

# Plot the results
Predictor_instance.plot_prob_forecasts(test_ds = test_ds)

















