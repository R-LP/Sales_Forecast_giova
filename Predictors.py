from data_processing import *
import gluonts


def launch_data():
        Transactions_obj = TransactionsMonthlyGranular("Lancome_sub_sub.csv")
        data = Transactions_obj.Product_sales(["UVEX 2019 BRIGHT SPF50 PA4 30ML FG", "REN M LIFT NIGHT CR J50ML ASIA"], "day")
        print(data)
        data["SalesQuantity"].plot()
        plt.grid(which="both")
        plt.show()



from gluonts.dataset.common import ListDataset
train_ds = ListDataset([{'start': data.OrderDate.iloc[0], 'target': data.SalesQuantity}], freq = 'D')


estimator = gluonts.model.simple_feedforward.SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=len(test_series) - len(train_series),
    context_length=365,
    freq='D',
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=200,),)

