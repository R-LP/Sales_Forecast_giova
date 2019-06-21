from data_processing import *




Transactions_obj = TransactionsMonthlyGranular("Lancome_sub_sub.csv")


Transactions_obj.groupby_product("week")

print(Transactions_obj.data)






























