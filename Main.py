import warnings
warnings.filterwarnings("ignore")
from Predictors import *


prediction_length = 40
train_ds, test_ds = train_test_set(min_date = "2016-07-01", max_date = "2019-06-30",
                                                        prediction_length = prediction_length)


predictor = train_predictor(freq = "D", prediction_length = prediction_length, train_ds = train_ds)
forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor = predictor, num_eval_samples = 100)
forecasts = list(forecast_it)
tss = list(ts_it)
plot_forecasts(tss, forecasts, past_length = 200, num_plots = 2, train_ds = train_ds)










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















