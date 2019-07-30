import warnings
warnings.filterwarnings("ignore")
from Predictors import *

# For prediction
prediction_length = 303
freq = "D"

# For neural network architecture
epochs = 200
num_layers = 3 
batch_size = 64 


# Draw test and train sets
train_ds, test_ds = train_test_set(min_date = "2016-07-01", max_date = "2019-06-30",
                                                        prediction_length = prediction_length, freq = freq)

# Train the model
predictor = train_predictor(freq = freq, prediction_length = prediction_length, train_ds = train_ds, epochs = epochs, num_layers = num_layers, batch_size = batch_size)

# Make the forecasts
forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor = predictor, num_eval_samples = 100)

# Plot the prediction
forecasts = list(forecast_it)
tss = list(ts_it)
forecastss = pd.DataFrame(forecasts)
tsss = pd.DataFrame(tss)
forecastss.to_csv("C:/Users/jean.debeney/OneDrive - Ekimetrics/Downloads/forecasts.csv")
tsss.to_csv("C:/Users/jean.debeney/OneDrive - Ekimetrics/Downloads/tss.csv")
plot_forecasts(tss, forecasts, past_length = 100, num_plots = 2, train_ds = train_ds)

















