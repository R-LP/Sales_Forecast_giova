import warnings
warnings.filterwarnings("ignore")
from Predictors import *
from data_processing import *
import copy


## Uploading the data
Transactions_obj = TransactionsData(data, promo_data)


######## Automated selection for best fit algorithm - Evaluating the performance on the test set
if algorithm=='All':

    dict_mse_algo = {}

    ## DeepAR
    Transactions_obj.algorithm = 'DeepAR'
    train_ds, test_ds = Transactions_obj.train_predict_sets(list_products = list_products, type_split = 'train_test', prediction_length = prediction_length, 
                                                            min_date = min_date, max_date = max_date, freq = freq)
    Predictor_instance_test = Predictor_sales()
    Predictor_instance_test.algorithm = 'DeepAR'
    Predictor_instance_test.define_DeepAR_predictor(freq = freq, prediction_length = prediction_length, epochs=epochs, num_layers = num_layers, batch_size = batch_size)
    Predictor_instance_test.train_predictor(train_ds = train_ds)
    forecast_it, ts_it = Predictor_instance_test.make_predictions(test_ds)
    Predictor_instance_test.save_csv("test", forecast_it, ts_it)
    ts_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'ts test.csv'))
    forecast_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'forecast test.csv'))
    mse_df = Predictor_instance_test.mse_compute(forecast_test, ts_test)
    dict_mse_algo['DeepAR'] = mse_df['MSE'].mean()
    print('mean_mse_DeepAR:', '%0.f' %dict_mse_algo['DeepAR'])

    ##Prophet
    Transactions_obj.algorithm = 'Prophet'
    train_ds, test_ds = Transactions_obj.train_predict_sets(list_products = list_products, type_split = 'train_test', prediction_length = prediction_length, 
                                                            min_date = min_date, max_date = max_date, freq = freq)
    Predictor_instance_test = Predictor_sales()
    Predictor_instance_test.algorithm = 'Prophet'
    Predictor_instance_test.define_Prophet_predictor(freq = freq, prediction_length = prediction_length, prophet_params = prophet_params)
    forecast_it, ts_it = Predictor_instance_test.make_predictions(test_ds)
    Predictor_instance_test.save_csv("test", forecast_it, ts_it)
    ts_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'ts test.csv'))
    forecast_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'forecast test.csv'))
    mse_df = Predictor_instance_test.mse_compute(forecast_test, ts_test)
    dict_mse_algo['Prophet'] = mse_df['MSE'].mean()
    print('mean_mse_Prophet:', '%0.f' %dict_mse_algo['Prophet'])

    ##ARIMA
    Transactions_obj.algorithm = 'ARIMA'
    train_ds, test_ds = Transactions_obj.train_predict_sets(list_products = list_products, type_split = 'train_test', prediction_length = prediction_length, 
                                                            min_date = min_date, max_date = max_date, freq = freq)
    Predictor_instance_test = Predictor_sales()
    Predictor_instance_test.algorithm = 'ARIMA'
    forecast_it, ts_it = Predictor_instance_test.make_predictions(test_ds)
    Predictor_instance_test.save_csv("test", forecast_it, ts_it)
    ts_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'ts test.csv'))
    forecast_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'forecast test.csv'))
    mse_df = Predictor_instance_test.mse_compute(forecast_test, ts_test)
    dict_mse_algo['ARIMA'] = mse_df['MSE'].mean()
    print('mean_mse_ARIMA:', '%0.f' %dict_mse_algo['ARIMA'])

    algorithm = min(dict_mse_algo, key=dict_mse_algo.get)
    print('Best algo is:', algorithm)


######## Testing on specified algorithm
else:
    ## Creating train and test sets out of the transaction/promo data object
    train_ds, test_ds = Transactions_obj.train_predict_sets(list_products = list_products, type_split = 'train_test', prediction_length = prediction_length, 
                                                            min_date = min_date, max_date = max_date, freq = freq)
    
    ## Init instance
    Predictor_instance_test = Predictor_sales()

    if algorithm=='DeepAR':

        ## Get models structure
        Predictor_instance_test.define_DeepAR_predictor(freq = freq, prediction_length = prediction_length, epochs=epochs, num_layers = num_layers, batch_size = batch_size)
        ## Training predictor object
        Predictor_instance_test.train_predictor(train_ds = train_ds)

    elif algorithm=='Prophet':
        ## Get model structure
        Predictor_instance_test.define_Prophet_predictor(freq = freq, prediction_length = prediction_length, prophet_params = prophet_params)

    elif algorithm=='ARIMA':
        print('ARIMA modeling')

    ## Training predictor object & Computing forecasts
    forecast_it, ts_it = Predictor_instance_test.make_predictions(test_ds)
    forecast_plot = copy.deepcopy(forecast_it)
    ts_plot = copy.deepcopy(ts_it)
    ## Saving forecasts into csv file
    Predictor_instance_test.save_csv("test", forecast_it, ts_it)
    ## Computing mse
    ts_test = pd.read_csv(os.path.join(OUTPUT_FOLDER,  "ts " +"_"+ str(min_date) +"_"+ str(max_date) +"_"+ str(algorithm) +"_"+ str(freq) +"_"+ 'test' + ".csv"))
    forecast_test = pd.read_csv(os.path.join(OUTPUT_FOLDER, "forecast " +"_"+ str(min_date) +"_"+ str(max_date) +"_"+ str(algorithm) +"_"+ str(freq) +"_"+ 'test' + ".csv"))
    mse_df = Predictor_instance_test.mse_compute(forecast_test, ts_test)
    mean_mse = mse_df['MSE'].mean()
    print('mean_mse:', '%0.f' %mean_mse)
    ## Plotting results - From deepcopy forecast objects
#     Predictor_instance_test.plot_prob_forecasts(forecast_plot, ts_plot)



# ######## Final prediction of censored future data

# ## Init instance
# Predictor_instance_final = Predictor_sales()
# Predictor_instance_final.algorithm = algorithm

# ## Creating train/future sets out of the transaction/promo data object
# Transactions_obj.algorithm = algorithm
# train_final_ds, future_ds = Transactions_obj.train_predict_sets(list_products = list_products, type_split = 'total_future', prediction_length = prediction_length, 
#                                                                 min_date = min_date, max_date = max_date, freq = freq)

# if algorithm=='DeepAR':
#     ## Get model structure
#     Predictor_instance_final.define_DeepAR_predictor(freq = freq, prediction_length = prediction_length, epochs=epochs, num_layers = num_layers, batch_size = batch_size)

#     ## Training predictor object
#     Predictor_instance_final.train_predictor(train_ds = train_final_ds)

# elif algorithm=='Prophet':
#     ## Get model structure
#     Predictor_instance_final.define_Prophet_predictor(freq = freq, prediction_length = prediction_length, prophet_params = prophet_params)

# elif algorithm=='ARIMA':
#     print('ARIMA modeling')

# ## Training predictor object & Computing forecasts
# forecast_future_it, ts_future_it = Predictor_instance_final.make_predictions(future_ds)
# forecast_future_plot = copy.deepcopy(forecast_future_it)
# ts_future_plot = copy.deepcopy(ts_future_it)

# ## Saving forecasts into csv file
# Predictor_instance_final.save_csv("future", forecast_future_it, ts_future_it)

# ## Plotting the results - 
# Predictor_instance_final.plot_prob_forecasts(forecast_future_plot, ts_future_plot)