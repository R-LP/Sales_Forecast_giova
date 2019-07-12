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
,"BEX12 PURIG FOAM 125ML JP"
,"GENIFIQUE EYE CARE J15ML"
,"REN M LIFT EYES CR J15ML ASIA"
,"REN M LIFT DAY CREAM 50ML"
,"REN M LIFT NIGHT CR J50ML ASIA"
,"ADVANCED GENIFIQUE SENSITIVE B20ML"
,"REN FLASH LIFT LOT 2016 400ML"
,"REN FLASH LIFT LOT 2016 200ML"
,"HZ NEOCALM AQUA GEL P/B200M ASIA"
,"RENG12 M-LIFT EMULSION 100ML JP"
,"GRANDIOSE WP 01"
,"HYDRAZEN NEOCALM CR TTP J50M ASIA"
,"CONF TONIQUE B200ML"
,"HYDRAZEN NEOCALM CR GEL J50ML ASIA"
,"APC EYE CREAM J20ML ASIA"
,"REN M LIFT PLASMA FL50ML ASIE"
,"UVEX 2017 AQ GEL SPF50 PA4 30ML FG"
,"GRANDIOSE SP 01 ASIA"
,"UVEX 2017 BB1 SPF50 PA4 30ML FG"
,"CONF CREAM MOUSSE T125ML"
,"HYDRAZEN NC FLUIDE V.MOIST 100ML JP"
,"CONFORT TONIQUE 400ML"
,"HYDRAZEN NIGHT J50ML ASIA"
,"ABSOLUE PC LOTION B150ML ASIA"
,"GENIFIQUE 13 SERUM B30ML"
,"RML FULL SPEC CREAM J50ML ASIE"
,"BEX SPOTERASER 2017 30ML QD"
,"BEX DAY CREAM 2016 50ML"
,"ABSOLUE BX 13 EYES J20ML ASIA"]


def train_test_set(min_date, max_date, prediction_length):
        Transactions_obj = TransactionsMonthlyGranular("Lanc_alg2.csv")
        data = Transactions_obj.Product_sales(list_products, "week", min_date,  max_date)
        #print(data)
        train = data[:-prediction_length]
        test = data[-prediction_length:]
        train_ds = ListDataset([{'start': train.OrderDate.index[0], "target": data.SalesQuantity}], freq = 'W')
        test_ds = ListDataset([{"start": test.OrderDate.index[0], "target": data.SalesQuantity}], freq = "W")
        return train_ds, test_ds


# freq = "D"
# prediction_length = 15
def train_predictor(freq, prediction_length, train_ds):
        estimator = DeepAREstimator(freq=freq, prediction_length=prediction_length,
                                        trainer=Trainer(epochs=150))
        predictor = estimator.train(training_data=train_ds)
        return predictor


def plot_forecasts(tss, forecasts, past_length, num_plots):
        for target, forecast in islice(zip(tss, forecasts), num_plots):
                ax = target[-past_length:].plot(figsize = (12,5), linewidth=2)
                forecast.plot(color = 'g')
                plt.grid(which = 'both')
                plt.legend(["observations", "median prediction", 
                    "90% confidence interval", "50% confidence interval"])
                plt.show()















