from data_processing import *


data_obj = Data()
#data_obj.data = read_from_transactions_folder("Lancome_sub_sub.csv")


data_obj.data = data_obj.read_from_transactions_folder("Lancome_sub_sub.csv")
print(data_obj.data)










































