from data_processing import *
import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from itertools import islice

list_products = ["CONF TONIQUE B400ML"
,"GENIFIQUE 13 SERUM B75ML"
,"ADV GEN EYES LIGHT PEARLB20ML"
,"GENIFIQUE ADV EYE CARE J15ML"
,"ADVANCED GENIFIQUE SENSITIVE B20ML"
,"REN M LIFT NIGHT CR J50ML ASIA"
,"REN M LIFT DAY CREAM 50ML"
,"REN M LIFT EYES CR J15ML ASIA"
,"BEX12 PURIG FOAM 125ML JP"
,"REN FLASH LIFT LOT 2016 400ML"
,"REN FLASH LIFT LOT 2016 200ML"
,"GENIFIQUE EYE CARE J15ML"
,"RENG12 M-LIFT EMULSION 100ML JP"
,"APC EYE CREAM J20ML ASIA"
,"HYDRAZEN NEOCALM CR TTP J50M ASIA"
,"HZ NEOCALM AQUA GEL P/B200M ASIA"
,"RML FULL SPEC CREAM J50ML ASIE"
,"REN M LIFT PLASMA FL50ML ASIE"
,"HYDRAZEN NEOCALM CR GEL J50ML ASIA"
,"RML FULL SPECTRUM SRM P/B50ML ASIE"]


def train_test_set(min_date, max_date):
        Transactions_obj = TransactionsMonthlyGranular("Lanc_sub_sub.csv")
        data = Transactions_obj.Product_sales(list_products, "day", min_date,  max_date)
        #print(data)
        train = data[:-15]
        test = data[-15:]
        train_ds = ListDataset([{'start': train.OrderDate.index[0], "target": data.SalesQuantity}], freq = 'D')
        test_ds = ListDataset([{"start": test.OrderDate.index[0], "target": data.SalesQuantity}], freq = "D")
        return train_ds, test_ds






        '''
        estimator = DeepAREstimator(freq="D", prediction_length=30, trainer=Trainer(epochs=10))
        predictor = estimator.train(training_data=train_ds)
        
        forecast_it, ts_it = make_evaluation_predictions(
                        dataset=test_ds,  # test dataset
                        predictor=predictor,  # predictor
                        num_eval_samples=100,  # number of sample paths we want for evaluation
        )

        egg_metrics, item_metrics = Evaluator()(
                ts_it, forecast_it, num_series = len(test_ds)
        )

        print(egg_metrics)
        '''
        '''
        for test_entry, forecast in zip(test_ds, predictor.predict(test_ds)):
                to_pandas(test_entry)[-150:].plot(linewidth=2)
                forecast.plot(color = 'g')
        plt.grid(which = 'both')
        plt.show()
        '''

