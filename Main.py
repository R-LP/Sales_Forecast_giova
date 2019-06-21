from data_processing import *




Transactions_obj = TransactionsMonthlyGranular("Lancome_sub_sub.csv")
#Transactions_obj.groupby_product("month")

Transactions_obj.Product_sales("UVEX 2019 BRIGHT SPF50 PA4 30ML FG", "day")
print(Transactions_obj.data)






























