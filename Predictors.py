from data_processing import *
import gluonts
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from itertools import islice
from pathlib import Path
from gluonts.model.predictor import Predictor



list_products = ["CONF TONIQUE B400ML",
"GENIFIQUE 13 SERUM B75ML",
"ADV GEN EYES LIGHT PEARLB20ML"
]


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


    def define_DeepAR_predictor(self, freq, prediction_length, train_ds, epochs, num_layers, batch_size):
        self.predictor = DeepAREstimator(freq=freq, prediction_length=prediction_length,
                                trainer=Trainer(ctx="cpu", epochs=epochs, batch_size = batch_size, num_batches_per_epoch = 100), num_layers = num_layers)


    def train_predictor(self, train_ds):
        self.predictor = self.predictor.train(training_data=train_ds)
        return self.predictor


    def make_predictions(self, test_ds):
        forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor = self.predictor, num_eval_samples = 100)
        return forecast_it, ts_it


    def plot_prob_forecasts(self, test_ds):
        forecast_plot, ts_plot = make_evaluation_predictions(dataset = test_ds, predictor = self.predictor, num_eval_samples = 100)
        tss = list(ts_plot)
        forecasts = list(forecast_plot)
        ts_entry = tss[0]
        forecast_entry = forecasts[0]
        plot_length = 150 
        prediction_intervals = (50.0, 90.0)
        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
        forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")
        plt.show()


    def save_csv(self, name, forecast_csv, ts_csv):
        ts_csv = list(ts_csv)
        forecast_csv = list(forecast_csv)
        ts_csv = ts_csv[0]
        forecast_csv = forecast_csv[0]
        forecast_csv = pd.DataFrame(forecast_csv.mean)
        ts_csv = pd.DataFrame(ts_csv)
        forecast_name = "forecast " + name + ".csv"
        ts_name = "ts " + name + ".csv"
        forecast_csv.to_csv(os.path.join(OUTPUT_FOLDER, forecast_name))
        ts_csv.to_csv(os.path.join(OUTPUT_FOLDER, ts_name))



    # For theDeepAR Estimator
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


