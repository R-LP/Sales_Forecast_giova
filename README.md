# Sales Forecast SKU granularity

This tool enables to generate sales forecasts from transcastion historical data. Predictions can also be enhanced by additional external data like promotional/marketing data and event related data. 

## Getting Started

Follow this instructions before trying to use the tool.

### Prerequisites

To get started, please set up a virtual environment using this command line in cmd prompt:
```
python3 -m venv SalesForecastEnv-env
```

Then activate the environment running:
```
SalesForecastEnv-env\Scripts\activate.bat
```

And import all the requirements from the .txt file with:
```
pip install -r requirements.txt
```

### Directories

Set up input and output directories'locations in file *forecast_env.env*

Then input the products'names you want to analyse. The input list can be a list of products or a list of lists of products. Aggregations will thus be computed at the list level to study forecasts on groups of products.


## Specify your prediction settings

Open the *settings.py* file. You need to specify the name of your data file, the time frame you are working on and the which algorithm you want to use. 
The available algorithms are ARIMA, DeepAR and Prophet. You may also use All algorithms which computes the results of the 3 algo on the test set and picks the best predictor for the final computation of the forecast.

* Complete the **Settings** section with the specifics of you data.

* Choose the prediction frame parameters

* Hyperparameters aren't supposed to be modified but you can play if it improves your prediction


## Run the predictions

To run compute the model and run predictions, run the *Main.py* file. 
Whatever the chosen algorithm, the process is to:

* Import the data
* Build train set as df[:-prediction_length] and test sets as df[-prediction_length:]

Then:
* Create a predictor instance, train and compute performance on the **test set**
* Plot test predictions
* Save test predictions in output_folder as .csv (ts as actual values, forecast as predicted values)

Then:
* Create a predictor instance, train and compute performance on the **training set**
* Plot train predictions 
* Save train predictions in output_folder as .csv (ts as actual values, forecast as predicted values)

You can remove plots or saving by passing code parts as comments.

