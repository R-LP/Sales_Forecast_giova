from data_processing import *
import warnings
warnings.filterwarnings("ignore")



Transactions_obj = TransactionsMonthlyGranular("Lancome_sub_sub.csv")
#Transactions_obj.groupby_product("month")

data = Transactions_obj.Product_sales(["UVEX 2019 BRIGHT SPF50 PA4 30ML FG", "REN M LIFT NIGHT CR J50ML ASIA"], "month")
print(data)






























