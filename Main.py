import warnings
warnings.filterwarnings("ignore")
from Predictors import *

# For prediction
prediction_length = 30
freq = "D"

# For neural network architecture
epochs = 200
num_layers = 4
batch_size = 16


# Draw test and train sets
train_ds, test_ds = train_test_set(min_date = "2016-07-01", max_date = "2019-05-20",
                                                        prediction_length = prediction_length, freq = freq)

# Train the model
predictor = train_predictor(freq = freq, prediction_length = prediction_length, train_ds = train_ds, epochs = epochs, num_layers = num_layers, batch_size = batch_size)

# Make the forecasts
forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor = predictor, num_eval_samples = 100)

# Plot the prediction
forecasts = list(forecast_it)
tss = list(ts_it)
plot_forecasts(tss, forecasts, past_length = 100, num_plots = 2, train_ds = train_ds)


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas

dataset = get_dataset("m4_hourly", regenerate=True)
train_entry = next(iter(dataset.train))
train_entry.keys()

test_entry = next(iter(dataset.test))
test_entry.keys()

test_series = to_pandas(test_entry)
train_series = to_pandas(train_entry)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r') # end of train dataset
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.show()

'''











#Transactions_obj = TransactionsMonthlyGranular("Lanc_alg.csv")
#Transactions_obj.groupby_product("month")

#data = Transactions_obj.Product_sales(["UVEX 2019 BRIGHT SPF50 PA4 30ML FG", "REN M LIFT NIGHT CR J50ML ASIA"], "day")
#print(data)

'''
dataset_alvin = pd.read_csv("C:\\Users\\jean.debeney\\OneDrive - Ekimetrics\\Documents\\Stock_forecast\\Alvin_rich_again_amended.csv")
train_alvin = dataset_alvin[:-30]
test_alvin = dataset_alvin[-30:]
train_ds = ListDataset([{'start': train_alvin.Date.index[0], "target": train_alvin.Close}], freq = "D")
test_ds = ListDataset([{'start': test_alvin.Date.index[0], "target": test_alvin.Close}], freq = "D")
'''















