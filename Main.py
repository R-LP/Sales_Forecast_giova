import Predictors
import warnings
warnings.filterwarnings("ignore")



#Transactions_obj = TransactionsMonthlyGranular("Lanc_alg.csv")
#Transactions_obj.groupby_product("month")

#data = Transactions_obj.Product_sales(["UVEX 2019 BRIGHT SPF50 PA4 30ML FG", "REN M LIFT NIGHT CR J50ML ASIA"], "day")
#print(data)

Predictors.prediction(min_date = "2017-07-01", max_date = "2019-06-30")


















