from data_processing import *
import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas

def launch_data():
        Transactions_obj = TransactionsMonthlyGranular("Lancome_sub.csv")
        data = Transactions_obj.Product_sales(["UVEX 2019 BRIGHT SPF50 PA4 30ML FG"], "day")
        print(data)
        #data["SalesQuantity"].plot()
        #plt.grid(which="both")
        #plt.show()
        train_ds = ListDataset([{'start': data.OrderDate.index[0], "target": data.SalesQuantity[:-30]}], freq = 'D')
        test_ds = ListDataset([{"start": data.OrderDate.index[0], "target": data.SalesQuantity[:-30]}], freq = "D")
        estimator = DeepAREstimator(freq="D", prediction_length=12, trainer=Trainer(epochs=10))
        predictor = estimator.train(training_data=train_ds)
        for test_entry, forecast in zip(test_ds, predictor.predict(test_ds)):
                to_pandas(test_entry)[-150:].plot(linewidth=2)
                forecast.plot(color = 'g')
        plt.grid(which = 'both')
        plt.show()



