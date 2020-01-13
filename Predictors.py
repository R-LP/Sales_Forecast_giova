from settings import *
from data_processing import *
import gluonts
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.prophet import ProphetPredictor
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from itertools import islice
from pathlib import Path
from gluonts.model.predictor import Predictor
import datetime 
from sklearn.metrics import mean_squared_error, mean_absolute_error



class Predictor_sales(object):
    def __init__(self, freq = "D", prediction_length = 30, epochs = 50, batch_size = 16, num_batches_per_epoch = 100, num_layers = 4):
        self.predictor = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=Trainer(ctx="cpu",
                            epochs=epochs,
                            batch_size = batch_size,
                            num_batches_per_epoch = num_batches_per_epoch),
            num_layers = num_layers)


    def define_DeepAR_predictor(self, freq, prediction_length, epochs, num_layers, batch_size):
        self.predictor = DeepAREstimator(freq=freq, prediction_length=prediction_length, context_length = prediction_length,
                                trainer=Trainer(ctx="cpu", epochs=epochs, batch_size = batch_size, num_batches_per_epoch = 100), num_layers = num_layers)


    def define_Prophet_predictor(self, freq, prediction_length , prophet_params):
        self.predictor = ProphetPredictor(freq = freq, prediction_length = prediction_length, prophet_params = prophet_params)


    def train_predictor(self, train_ds):
        self.predictor = self.predictor.train(training_data=train_ds)
        return self.predictor


    def make_predictions(self, eval_ds):
        if algorithm=='DeepAR':
            forecast_it, ts_it = make_evaluation_predictions(eval_ds, predictor = self.predictor, num_samples = 100)
            #return forecast_it, ts_it
        elif algorithm=='Prophet':
            forecast_it = self.predictor.predict(eval_ds)
            ts_it = []
            for p in range(len(list_products)):
                ts_it.append(pd.DataFrame({0:eval_ds.list_data[p]['target']}, 
                                       index=pd.date_range(min_date, periods=len(eval_ds.list_data[p]['target']), freq = freq, tz = None)))
        return forecast_it, ts_it



    def plot_prob_forecasts(self, eval_ds):
        if len(list_products)!=1:
            print('Which product no?')
            p = int(input({key: value for (key, value) in enumerate(list_products)}))
        else:
            p = 0
        forecast_plot, ts_plot = make_evaluation_predictions(dataset = eval_ds, predictor = self.predictor, num_samples = 100)
        tss = list(ts_plot)
        forecasts = list(forecast_plot)
        ts_entry = tss[p] # we plot only the first time serie to forecast
        forecast_entry = forecasts[p]
        plot_length = 70 
        prediction_intervals = (50.0, 90.0)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        _, ax = plt.subplots(1, 1, figsize=(10, 7))
        pd.plotting.register_matplotlib_converters()
        ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
        forecast_entry.plot(prediction_intervals=prediction_intervals, color='b')
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.show()

    # Run saving function before plotting anything
    def save_csv(self, name, forecast_it, ts_it):
        ts_name = "ts " + name + ".csv"
        forecast_name = "forecast " + name + ".csv"

        if len(list_products)!=1:
            forecast_csv = list(forecast_it)
            ts_csv = list(ts_it)
            forecast_entry = []
            ts_entry = []
            for p in range(len(list_products)):
                forecast_entry.append(forecast_csv[p].mean)
                ts_entry.append(ts_csv[p])
            forecast_csv = pd.DataFrame(data=np.array(forecast_entry).transpose(), 
                                        columns=list_products, 
                                        index=pd.date_range(forecast_csv[0].start_date, periods=prediction_length, freq=freq))
            forecast_csv = forecast_csv.rename_axis('OrderDate').reset_index()
            forecast_csv.to_csv(os.path.join(OUTPUT_FOLDER, forecast_name), index=False)
            for p in range(len(ts_entry)):
                if p==0:
                    ts_csv = ts_entry[0]
                else:
                    ts_csv = ts_csv.join(ts_entry[p], rsuffix=p)
            ts_csv.columns = list_products
            ts_csv = ts_csv.rename_axis('OrderDate').reset_index()
            ts_csv.to_csv(os.path.join(OUTPUT_FOLDER, ts_name), index=False)

        else:
            forecast_entry = list(forecast_it)[0]
            ts_entry = list(ts_it)[0]
            forecast_csv = pd.Series(forecast_entry.mean, index=pd.date_range(forecast_entry.start_date, periods=prediction_length, freq=freq), name=list_products[0])
            forecast_csv = forecast_csv.rename_axis('OrderDate').reset_index()
            forecast_csv.to_csv(os.path.join(OUTPUT_FOLDER, forecast_name), index=False)
            ts_csv = ts_entry.rename_axis('OrderDate')[0].rename(list_products[0]).reset_index()
            ts_csv.to_csv(os.path.join(OUTPUT_FOLDER, ts_name), index=False)

    # mse computation on test data
    def mse_compute(self, forecast_csv, ts_csv):
        ts_csv = ts_csv.loc[ts_csv['OrderDate'].isin(forecast_csv['OrderDate'])]
        for p in range(3):
            mse_product = mean_squared_error(ts_csv[list_products[p]], forecast_csv[list_products[p]])
            print(list_products[p], ':', "%.f" % mse_product)

    # For theDeepAR Estimator - not in use so far
    def create_transformation(self, freq, context_length, prediction_length):
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


