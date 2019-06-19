from data_processing import *

data = data_processing.read_from_transactions_folder("Lancome_sub_sub.csv")

data = Data(data)


periods = get_period_list(freq = 'D')
print(periods)









































