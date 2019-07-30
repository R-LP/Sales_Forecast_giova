from data_processing import *
import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from itertools import islice
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    Chain,
    FieldName,
    #ExpectedNumInstanceSampler,
    #FieldName,
    #InstanceSplitter,
    #SetFieldIfNotPresent,
)


#list_products = ["GENIFIQUE 13 SERUM B30ML", "REN FRENCH LIFT P50ML"]

def create_transformation(freq, context_length, prediction_length):
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
        ]
    )


def train_test_set(min_date, max_date, prediction_length, freq):
        # Load and process transactions data
        Transactions_obj = TransactionsMonthlyGranular("LAN.csv")
        data = Transactions_obj.Product_sales(list_products, "day", min_date,  max_date)
        ext_data = pd.read_csv("P:\\0. R&D\\6. SKU sales forecast\\1_Raw data\\Promo_data.csv")
        # Reading external data
        ext_data["OrderDate"] = pd.to_datetime(ext_data["OrderDate"])
        data = data.merge(ext_data, how = 'left', on = "OrderDate")
        # Creating different fields for the training dataset
        [f"FieldName.{k} = '{v}'" for k, v in FieldName.__dict__.items() if not k.startswith('_')]
        train = data[:-prediction_length]
        print(train)
        '''
        feat_static_cat: static (over time) categorical features, list with dimension equal to the number of features
        feat_static_real: static (over time) real features, list with dimension equal to the number of features
        feat_dynamic_cat: dynamic (over time) categorical features, array with shape equal to (target length, number of features)
        feat_dynamic_real: dynamic (over time) real features, array with shape equal to (target length, number of features)
        '''
        train_ds = ListDataset([{FieldName.TARGET: train.SalesQuantity, 
                                FieldName.START: train.OrderDate.index[0],
                                FieldName.FEAT_DYNAMIC_REAL: train[["Month", "Weekday", "Day", "Week"]],
                                }], 
                                freq=freq)

        test_ds = ListDataset([{FieldName.TARGET: data.SalesQuantity, 
                        FieldName.START: data.OrderDate.index[0],
                        FieldName.FEAT_DYNAMIC_REAL: data[["Month", "Weekday", "Day", "Week"]],
                        }], 
                        freq=freq)
        transformation = create_transformation(freq = freq, context_length = len(data.OrderDate) - prediction_length,  prediction_length = prediction_length)
        #train_tf = transformation(iter(train_ds), is_train = True)
        #test_tf = transformation(iter(test_ds), is_train = False)
        return train_ds, test_ds


def load_predictor():
        predictor_deserialized = Predictor.deserialize(Path("P:/0. R&D/6. SKU sales forecast/4_Predictors/"))
        return predictor_deserialized


def save_predictor(predictor):
    predictor.serialize(Path("P:/0. R&D/6. SKU sales forecast/4_Predictors/"))


def train_predictor(freq, prediction_length, train_ds, epochs, num_layers, batch_size):
        estimator = DeepAREstimator(freq=freq, prediction_length=prediction_length,
                                        trainer=Trainer(ctx="cpu", epochs=epochs, batch_size = batch_size, num_batches_per_epoch = 100), num_layers = num_layers)
        predictor = estimator.train(training_data=train_ds)
        return predictor


def plot_forecasts(tss, forecasts, past_length, num_plots, train_ds):
        evaluator = Evaluator(quantiles=[0.5], seasonality=2019)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(train_ds))
        print(agg_metrics)
        for target, forecast in islice(zip(tss, forecasts), num_plots):
                ax = target[-past_length:].plot(figsize = (12,5), linewidth=2)
                forecast.plot(color = 'g')
                plt.grid(which = 'both')
                plt.legend(["observations", "median prediction", 
                    "90% confidence interval", "50% confidence interval"])
                plt.show()


