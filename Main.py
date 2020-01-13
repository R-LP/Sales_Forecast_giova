import warnings
warnings.filterwarnings("ignore")
from Predictors import *
from data_processing import *


# Creating train and test sets out of the transaction/promo data object
Transactions_obj = TransactionsData(data, promo_data)

train_ds, test_ds = Transactions_obj.train_test_set(list_products = list_products, prediction_length = prediction_length, 
                                                    min_date = min_date, max_date = max_date, freq = freq)
train_final_ds, future_ds = Transactions_obj.train_future_set(list_products = list_products, prediction_length = prediction_length, 
                                                    min_date = min_date, max_date = max_date, freq = freq)

# Init instances
Predictor_instance_test = Predictor_sales()
Predictor_instance_final = Predictor_sales()

if algorithm=='DeepAR':

    # Get models structure
    Predictor_instance_test.define_DeepAR_predictor(freq = freq, prediction_length = prediction_length, epochs=2, num_layers = num_layers, batch_size = batch_size)
    Predictor_instance_final.define_DeepAR_predictor(freq = freq, prediction_length = prediction_length, epochs=2, num_layers = num_layers, batch_size = batch_size)

    # Training predictor objects
    Predictor_instance_test.train_predictor(train_ds = train_ds)
    Predictor_instance_final.train_predictor(train_ds = train_final_ds)

elif algorithm=='Prophet':

    # Get models structure
    Predictor_instance_test.define_Prophet_predictor(freq = freq, prediction_length = prediction_length, prophet_params = prophet_params)
    Predictor_instance_final.define_Prophet_predictor(freq = freq, prediction_length = prediction_length, prophet_params = prophet_params)

# Training predictor objects & Computing forecasts
forecast_it, ts_it = Predictor_instance_test.make_predictions(test_ds)
forecast_future_it, ts_future_it = Predictor_instance_final.make_predictions(future_ds)

# Saving forecasts into csv files
Predictor_instance_test.save_csv("test", forecast_it, ts_it)
Predictor_instance_final.save_csv("future", forecast_future_it, ts_future_it)

# Computing mse
ts_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'ts test.csv'))
forecast_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'forecast test.csv'))
Predictor_instance_test.mse_compute(forecast_test, ts_test)

# Plot the results
Predictor_instance_test.plot_prob_forecasts(test_ds)
Predictor_instance_final.plot_prob_forecasts(future_ds)