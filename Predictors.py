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
from pmdarima import auto_arima
import copy

## DeepAR, Prophet and ARIMA are trained differently on the data (see main.py file). Thus, the class functions varies depending on the algorithm to adapt to their training methods and output structure.


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
        self.algorithm = algorithm


    # DeepAR instance to be explicitly trained before predicting
    def define_DeepAR_predictor(self, freq, prediction_length, epochs, num_layers, batch_size):
        self.predictor = DeepAREstimator(freq=freq, prediction_length=prediction_length, context_length = prediction_length,
                                trainer=Trainer(ctx="cpu", epochs=epochs, batch_size = batch_size, num_batches_per_epoch = 100), num_layers = num_layers)


    # Prophet instance to implicitly trained during definition
    def define_Prophet_predictor(self, freq, prediction_length , prophet_params):
        self.predictor = ProphetPredictor(freq = freq, prediction_length = prediction_length, prophet_params = prophet_params)


    # ARIMA instance to implicitly trained during definition
    def train_ARIMA_predictor(self, eval_ds, p):
        return auto_arima(eval_ds.list_data[p]['target'][:-prediction_length], error_action='ignore', suppress_warnings=True, n_jobs=-1)


    def train_predictor(self, train_ds):
        self.predictor = self.predictor.train(training_data=train_ds)
        return self.predictor


    # Making predictions depend on the algo instance
    def make_predictions(self, eval_ds):
        if self.algorithm=='DeepAR':
            forecast_it, ts_it = make_evaluation_predictions(eval_ds, predictor = self.predictor, num_samples = 100)

        elif self.algorithm=='Prophet':
            train_ds = copy.deepcopy(eval_ds)
            for p in range(len(list_products)):
                train_ds.list_data[p]['target'] = train_ds.list_data[p]['target'][:-prediction_length]
            forecast_it = self.predictor.predict(train_ds)
            ts_it = []
            for p in range(len(list_products)):
                ts_it.append(pd.DataFrame({0:eval_ds.list_data[p]['target']}, 
                                       index=pd.date_range(min_date, periods=len(eval_ds.list_data[p]['target']), freq = freq, tz = None)))
        elif self.algorithm=='ARIMA':
            ts_it = []
            period_list_pred = pd.date_range(min_date, periods=len(eval_ds.list_data[0]['target']), freq = freq, tz = None)[-prediction_length:]
            for p in range(len(list_products)):
                AMIMA_predictor = self.train_ARIMA_predictor(eval_ds, p)
                pred = AMIMA_predictor.predict(n_periods=prediction_length)
                if p==0:
                    forecast_it = pd.DataFrame({'OrderDate':period_list_pred, 'Product':pred})
                else:
                    temp = pd.DataFrame({'OrderDate':period_list_pred, 'Product':pred})
                    forecast_it = forecast_it.merge(temp, on='OrderDate', how='left')
                forecast_it = forecast_it.rename(columns={'Product':list_products[p]})
                ts_it.append(pd.DataFrame({0:eval_ds.list_data[p]['target']}, 
                                       index=pd.date_range(min_date, periods=len(eval_ds.list_data[p]['target']), freq = freq, tz = None)))
            return forecast_it, ts_it
        return list(forecast_it), list(ts_it)


    # Plotting depends on the prediction output structure
    def plot_prob_forecasts(self, forecast_plot, ts_plot):
        if len(list_products)!=1:
            print('Which product no?')
            p = int(input({key: value for (key, value) in enumerate(list_products)}))
        else:
            p = 0
        if self.algorithm not in ['ARIMA']:
            ts_entry = ts_plot[p] # we plot only the first time serie to forecast
            forecast_entry = forecast_plot[p]
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

        else:
            history_plot_lenth = min(prediction_length*5, len(ts_plot[0]))
            ts_plot = ts_plot[p][-history_plot_lenth:].set_index(pd.DatetimeIndex(ts_plot[p][-history_plot_lenth:].index))
            forecast_plot = forecast_plot.set_index(pd.DatetimeIndex(forecast_plot['OrderDate'])).drop(columns=['OrderDate']).iloc[:,p]
            plt.figure(figsize=(10,6))
            plt.plot(ts_plot, color='C0', label='Observations')
            plt.plot(forecast_plot, color='b', label='Predictions')
            plt.legend()
            plt.show()


    # Run saving function before plotting anything
    def save_csv(self, name, forecast_it, ts_it):
        ts_name = "ts " + name + ".csv"
        forecast_name = "forecast " + name + ".csv"

        if self.algorithm not in ['ARIMA']:
            if len(list_products)!=1:
                forecast_entry = []
                for p in range(len(list_products)):
                    forecast_entry.append(forecast_it[p].mean)
                start_dt = pd.date_range(min_date, periods=len(ts_it[0]), freq = freq, tz = None)[-prediction_length]
                #print(start_dt)
                forecast_csv = pd.DataFrame(data=np.array(forecast_entry).transpose(), 
                                            columns=list_products, 
                                            index=pd.date_range(start_dt, periods=prediction_length, freq=freq))
                forecast_csv = forecast_csv.rename_axis('OrderDate').reset_index()
                forecast_csv.to_csv(os.path.join(OUTPUT_FOLDER, forecast_name), index=False)
                for p in range(len(list_products)):
                    if p==0:
                        ts_csv = ts_it[0]
                    else:
                        ts_csv = ts_csv.join(ts_it[p], rsuffix=p)
                ts_csv.columns = list_products
                ts_csv = ts_csv.rename_axis('OrderDate').reset_index()
                ts_csv.to_csv(os.path.join(OUTPUT_FOLDER, ts_name), index=False)

            else:
                forecast_entry = forecast_it[0]
                ts_entry = ts_it[0]
                forecast_csv = pd.Series(forecast_entry.mean, index=pd.date_range(forecast_entry.start_date, periods=prediction_length, freq=freq), name=list_products[0])
                forecast_csv = forecast_csv.rename_axis('OrderDate').reset_index()
                forecast_csv.to_csv(os.path.join(OUTPUT_FOLDER, forecast_name), index=False)
                ts_csv = ts_entry.rename_axis('OrderDate')[0].rename(list_products[0]).reset_index()
                ts_csv.to_csv(os.path.join(OUTPUT_FOLDER, ts_name), index=False)
           
        else: # For ARIMA
            forecast_it.to_csv(os.path.join(OUTPUT_FOLDER, forecast_name), index=False)
            if len(list_products)!=1:
                for p in range(len(list_products)):
                    if p==0:
                        ts_csv = ts_it[0]
                    else:
                        ts_csv = ts_csv.join(ts_it[p], rsuffix=p)
                ts_csv.columns = list_products
                ts_csv = ts_csv.rename_axis('OrderDate').reset_index()
                ts_csv.to_csv(os.path.join(OUTPUT_FOLDER, ts_name), index=False)

            else:
                ts_entry = ts_it[0]
                ts_csv = ts_entry.rename_axis('OrderDate')[0].rename(list_products[0]).reset_index()
                ts_csv.to_csv(os.path.join(OUTPUT_FOLDER, ts_name), index=False)



    # MSE computation on test data
    def mse_compute(self, forecast_csv, ts_csv):
        ts_csv = ts_csv.loc[ts_csv['OrderDate'].isin(forecast_csv['OrderDate'])]
        mse_products = []
        for p in range(len(list_products)):
            mse_products.append(mean_squared_error(ts_csv[list_products[p]], forecast_csv[list_products[p]]))
        mse_df = pd.DataFrame({'ProductEnglishname':list_products, 'MSE': mse_products})
        print(mse_df)
        return(mse_df)


    # For theDeepAR Estimator - NOT IN USE SO FAR - Supposed to build automated feature engineering on TS
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


