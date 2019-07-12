import warnings
warnings.filterwarnings("ignore")
from Predictors import *


#Transactions_obj = TransactionsMonthlyGranular("Lanc_alg.csv")
#Transactions_obj.groupby_product("month")

#data = Transactions_obj.Product_sales(["UVEX 2019 BRIGHT SPF50 PA4 30ML FG", "REN M LIFT NIGHT CR J50ML ASIA"], "day")
#print(data)


train_ds, test_ds = train_test_set(min_date = "2016-07-01", max_date = "2019-06-30",
                                                        prediction_length = 60)

predictor = train_predictor(freq = "W", prediction_length = 30, train_ds = train_ds)


forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor = predictor, num_eval_samples = 100)
forecasts = list(forecast_it)
tss = list(ts_it)
plot_forecasts(tss, forecasts, past_length = 100, num_plots = 2)

















