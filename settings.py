############################################### Settings ###############################################
from dotenv import load_dotenv
import os
import json
load_dotenv(os.path.join(os.getcwd(), "forecast_env.env"))
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER")
TRANSACTIONS_FOLDER = os.getenv("TRANSACTIONS_FOLDER")
PROMO_DATA_FOLDER = os.getenv("PROMO_DATA_FOLDER")
# list_products=json.loads(os.getenv("list_products"))


# Data file - .csv file name
data="corona_recovery_formated.csv"
# Mapping inputs of you data if the following columns names are not exactly the same - replace xxx, yyy, zzz with the date, the product key and the quantity column names respectively
# input_cols_mapping = {'xxx':'OrderDate',
#                      'yyy':'ProductEnglishname',
#                      'zzz':'SalesQuantity'}
input_cols_mapping = {'date':'OrderDate',
                      'variable':'Granulcolname',
                      'value':'SalesQuantity'}
#input_cols_promo_mapping = {'Date':'OrderDate'}

# Promo data file - .csv additional data file - Make sure OrderDate is the name of the date column - If no additional data, let it be None
promo_data=None

# Setting the name of the algorithm - choices ar: ARIMA, DeepAR, Prophet or All
algorithm = "Prophet"

# Prediction frame settings - Minimum historical date and Maximum historical date
prediction_length=10 # int
freq="D" # D,W,Y
min_date="2020-01-22" # "yyyy-mm-dd"
max_date="2020-02-16" # "yyyy-mm-dd"


###################################### Algorithms hyperparameters ######################################

# DeepAR hyperparameters
epochs=3
num_layers=3
batch_size=64

# Prophet hyperparameters
mcmc_samples=0
changepoint_prior_scale=0.01
interval_width=0.9
seasonality_mode='multiplicative'
yearly_seasonality=False
weekly_seasonality=False
daily_seasonality=True
regressor_list = []
prophet_params = {'mcmc_samples' : mcmc_samples, 'changepoint_prior_scale' : changepoint_prior_scale, 'interval_width' : interval_width,
                  'seasonality_mode' : seasonality_mode,
                  'weekly_seasonality' : weekly_seasonality,
                  'daily_seasonality' : daily_seasonality,
                  'yearly_seasonality' : yearly_seasonality,}

###################################### Product list to forecast ######################################
## Use "" for strings instead of '' - Can be a list of lists 
## Replace special characters by '_'

list_products = ['Mainland_China_Anhui', 'Mainland_China_Beijing',
       'Mainland_China_Chongqing', 'Mainland_China_Fujian',
       'Mainland_China_Gansu', 'Mainland_China_Guangdong',
       'Mainland_China_Guangxi', 'Mainland_China_Guizhou',
       'Mainland_China_Hainan', 'Mainland_China_Hebei',
       'Mainland_China_Heilongjiang', 'Mainland_China_Henan',
       'Mainland_China_Hubei', 'Mainland_China_Hunan',
       'Mainland_China_Inner_Mongolia', 'Mainland_China_Jiangsu',
       'Mainland_China_Jiangxi', 'Mainland_China_Jilin',
       'Mainland_China_Liaoning', 'Mainland_China_Ningxia',
       'Mainland_China_Qinghai', 'Mainland_China_Shaanxi',
       'Mainland_China_Shandong', 'Mainland_China_Shanghai',
       'Mainland_China_Shanxi', 'Mainland_China_Sichuan',
       'Mainland_China_Tianjin', 'Mainland_China_Tibet',
       'Mainland_China_Xinjiang', 'Mainland_China_Yunnan',
       'Mainland_China_Zhejiang', 'Thailand_nan', 'Japan_nan',
       'South_Korea_nan', 'Taiwan_Taiwan', 'US_Seattle__WA',
       'US_Chicago__IL', 'US_Tempe__AZ', 'Macau_Macau',
       'Hong_Kong_Hong_Kong', 'Singapore_nan', 'Vietnam_nan',
       'France_nan', 'Nepal_nan', 'Malaysia_nan', 'Canada_Toronto__ON',
       'Canada_British_Columbia', 'US_Orange__CA', 'US_Los_Angeles__CA',
       'Australia_New_South_Wales', 'Australia_Victoria',
       'Australia_Queensland', 'Cambodia_nan', 'Sri_Lanka_nan',
       'Germany_nan', 'Finland_nan', 'United_Arab_Emirates_nan',
       'Philippines_nan', 'India_nan', 'Canada_London__ON', 'Italy_nan',
       'UK_nan', 'Russia_nan', 'Sweden_nan', 'US_Santa_Clara__CA',
       'Spain_nan', 'Australia_South_Australia', 'US_Boston__MA',
       'US_San_Benito__CA', 'Belgium_nan', 'US_Madison__WI',
       'Others_Diamond_Princess_cruise_ship', 'US_San_Diego_County__CA',
       'US_San_Antonio__TX', 'Egypt_nan', 'China_total']

