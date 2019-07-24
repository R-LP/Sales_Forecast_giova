from data_processing import *
import gluonts
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from itertools import islice
from gluonts.transform import FieldName
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    Chain,
    #ExpectedNumInstanceSampler,
    #FieldName,
    #InstanceSplitter,
    #SetFieldIfNotPresent,
)



list_products = ["CONF TONIQUE B400ML",
"GENIFIQUE 13 SERUM B75ML",
"ADV GEN EYES LIGHT PEARLB20ML",
"GENIFIQUE ADV EYE CARE J15ML",
"ADVANCED GENIFIQUE SENSITIVE B20ML",
"REN M LIFT NIGHT CR J50ML ASIA",
"REN M LIFT DAY CREAM 50ML",
"REN M LIFT EYES CR J15ML ASIA",
"BEX12 PURIG FOAM 125ML JP",
"REN FLASH LIFT LOT 2016 400ML",
"REN FLASH LIFT LOT 2016 200ML",
"GENIFIQUE EYE CARE J15ML",
"RENG12 M-LIFT EMULSION 100ML JP",
"APC EYE CREAM J20ML ASIA",
"HYDRAZEN NEOCALM CR TTP J50M ASIA",
"HZ NEOCALM AQUA GEL P/B200M ASIA",
"RML FULL SPEC CREAM J50ML ASIE",
"REN M LIFT PLASMA FL50ML ASIE",
"HYDRAZEN NEOCALM CR GEL J50ML ASIA",
"RML FULL SPECTRUM SRM P/B50ML ASIE",
"CONF CREAM MOUSSE T125ML",
"BEX SPOTERASER 2017 30ML QD",
"ABSOLUE PC LOTION B150ML ASIA",
"GENIFIQUE SERUM FL100ML /NP",
"UVEX 2017 AQ GEL SPF50 PA4 30ML FG",
"UVEX 2017 BB1 SPF50 PA4 30ML FG",
"CONF TONIQUE B200ML",
"HYDRAZEN NC FLUIDE V.MOIST 100ML JP",
"HYDRAZEN NIGHT J50ML ASIA",
"APC CR YEUX J20ML ASIA",
"ABSOLUE PC UV SPF50 PA4 T30ML",
"HYDRAZEN GEL ESS P/B30ML",
"BEX EMULSION FRESH RENO 2016 100ML",
"BEX ESS IN LOTION 2017 150ML",
"BEX DAY CREAM 2016 50ML",
"ABSOLUE BX 13 EYES J20ML ASIA",
"BI FACIL EYE B125ML",
"DOUC GALATEIS P/B400ML",
"GENIFIQUE 13 SERUM B30ML",
"BEX NIGHT CREAM 2015 50ML",
"GENIFIQUE 13 SERUM B50ML",
"ADV GEN LIGHT PEARL LASHE B20ML /NF",
"ABSOLUE PC SERUM YEUX F/P15ML",
"UVEX 2019 AQ GEL SPF50 PA4 30ML FG",
"APC SOFT CREAM RECH P60ML",
"ECLAT GEL T125ML",
"APC SERUM YEUX F/P15ML",
"ABS PC WHITE AURA LOT B150ML",
"CONF TON DUO 400ML SG TKS 18 MAY",
"APC W AURA CR J50ML ASIA",
"ABSOLUE BX 13 CREAM J50ML ASIA",
"REN EYE CREAM SET XMAS17",
"GENIFIQUE SERUM B100ML RED OS",
"UVEX CC SPF50 S2 30ML ASIA",
"HZ DAY CR 50ML SET XMAS17",
"GENIFIQUE LP EYE SET XMAS17",
"ABS EXTRAIT SERUM F/P30ML",
"ABS PP NETT MOUSSE ATO150ML",
"SG_MAY19_CT TRIO SET",
"HYDRAZEN NIGHT MASK J75ML",
"ADV GENIFIQUE LIGHT PEARL EYE MASK",
"SG NOV18_REN ROUTINE DIS SET",
"ABSOLUE BX 13 NIGHT CR J75ML ASIA",
"UVEX 2019 TONE UP WH SPF50 30ML FG",
"ABS OIL P/B30ML",
"ABS EXT ELIXIR RECH P50ML",
"UVEX 2019 BRIGHT SPF50 PA4 30ML FG",
"ABS PP DEMAQ HUILE P200ML",
"DOUC GALATEIS P/B200ML",
"SG NOV18_GEN50 SEN DIS SET",
"GENIFIQUE MASK 16X6 JP",
"BEX DOUBLE ESSENCE 2018 30ML JP OS",
"APC ROSE MASK J75ML",
"APC MIDNIGHT BI PHASE OIL P/B15ML",
"ABS PC MASK J75ML",
"ABSOLUE L EXTRAIT J50ML",
"UVEX 2019 BB2 SPF50 PA4 30ML FG",
"HYDRAZEN EYES P/B15ML",
"GEN EYE SG TKS 18 MAY",
"GEN 75ML SG TKS 17 NOV",
"SG NOV18_GEN EYE LP DIS SET",
"UVEX CC SPF50 S3 30ML ASIA",
"GEN LP & YEUX 2FG SET XMAS17 10DIS",
"CONF TON 400ML SG TKS 17 NOV",
"APC WTE AURA CC P/B30ML",
"RML FULL SPECTR ULTRA ASIA J75ML OS",
"DOUC EAU MICELL P/B400ML",
"GK_GEN SENSITIVE SET 20ML_080917",
"REN DAY NT LOT SG TKS 18 MAY",
"MASQUE EN SUCRE CONFORT T100ML",
"GEN EYE SG TKS 17 NOV",
"SG NOV18_GEN50 EYE SET",
"GEN 50ML SEN SG TKS 18 MAY",
"REN DAY NT LOT 20 DIS SG TKS 17 NOV",
"CONF TON DUO 400ML SG TKS 18 MAY_V2",
"CONF GALATEE P/B200ML",
"REN CICA HEALING CREAM T50ML",
"GEN LP SG TKS 18 MAY",
"GEN 50ML EYE SG TKS 18 MAY",
"SG_MAY19_REN SPEC CR+NI CR SET_PH1",
"SG_MAY19_REN LOT+DAY+NIG SET_PH1",
"HDZ DAY NT LOT ESS GC SG TKS 18 MAY",
"SG_MAY19_GEN EYE CREAM _PH1",
"SG NOV18_HDZ DIS SET",
"CLARTE EXFOLIANCE T100ML",
"SG NOV18_ABS PC EYE SET",
"EDV DAY CREAM SET XMAS17",
"DOUC TONIQUE B200ML",
"SG NOV18_BEX LOT EMU NIG DIS SET",
"SG NOV18_REN SPEC CR SET",
"SG_REN_SPEC_CR_SERUM DIS SET",
"APC EYE & EYE SERUM SET XMAS17 10DIS",
"REN FRENCH LIFT P50ML",
"BEX12 MASK X6PC JP",
"EDV EXFOLIATING MASK J75ML",
"ABS CLEANSING FOAM T125ML",
"HDZ DAY NT LOT ESS GC 20DIS SG TKS 17NOV",
"RML CREAM J75ML ASIA 2017 OS",
"SG_MAY19_GEN75+EYE CR SET_PH1",
"SG_MAY19_GEN75+LP SET_PH1",
"GEN LP SG TKS 17 NOV",
"WDAY18_REN ULTRA SERUM+CREAM DISOUNT SET",
"REN DAY SG TKS 18 MAY",
"SG NOV18_GEN EYE CR SET",
"CNY 18 GEN SERUM 75ML + GEN SEN 20ML",
"SG NOV18_REN EYE SET",
"MASQUE CONFORT T100ML",
"SG NOV18_ABS EUM DAY ESS DIS SET",
"SG NOV18_GEN SEN SET",
"SGT_May19_Eye Cr+LP Set_PH2",
"SG NOV18_REN DANIEY DIS SET",
"ABS DAY CREAM SET XMAS18",
"ABSOLUE PC LIPS T15ML",
"REN DAY SG TKS 17 NOV",
"EDV EYE GEL T15ML",
"ABS DAY CREAM SET XMAS17",
"MASK HYDRA INT T100ML",
"REN EYE SG TKS 18 MAY",
"ABS EXT OIL P/B30ML /NP",
"GEN 50ML EYE SG TKS 17 NOV",
"CNY 18 GEN EYE CREAM & LP SET",
]

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
        ext_data["OrderDate"] = pd.to_datetime(ext_data["OrderDate"])
        data = data.merge(ext_data, how = 'left', on = "OrderDate")
        # Turning the dataset into a correct entry for the DeepAR algorithm        
        [f"FieldName.{k} = '{v}'" for k, v in FieldName.__dict__.items() if not k.startswith('_')]
        train = data[:-prediction_length]

        # Ability to add feat_static_cat (static categorical), feat_static_real (static real), feat_dynamic_cat, feat_dynamic_real (dyamic)
        train_ds = ListDataset([{FieldName.TARGET: train.SalesQuantity, 
                                FieldName.START: train.OrderDate.index[0],
                                FieldName.FEAT_DYNAMIC_REAL: train[["GWP", "VS", "year", "month", "week", "day", "dayofweek"]],
                                #FieldName.FEAT_STATIC_REAL: train[["month", "week", "day", "dayofweek"]],
                                }], 
                                freq=freq)

        test_ds = ListDataset([{FieldName.TARGET: data.SalesQuantity, 
                        FieldName.START: data.OrderDate.index[-prediction_length],
                        FieldName.FEAT_DYNAMIC_REAL: data[["GWP", "VS", "year", "month", "week", "day", "dayofweek"]],
                        #FieldName.FEAT_STATIC_REAL: data[["month", "week", "day", "dayofweek"]],
                        }], 
                        freq=freq)
        transformation = create_transformation(freq = freq, context_length = len(data.OrderDate) - prediction_length,  prediction_length = prediction_length)
        #train_tf = transformation(iter(train_ds), is_train = True)
        #test_tf = transformation(iter(test_ds), is_train = False)

        return train_ds, test_ds


def load_predictor():
        predictor_deserialized = Predictor.deserialize(Path("P:/0. R&D/6. SKU sales forecast/4_Predictors/"))
        return predictor_deserialized


def train_predictor(freq, prediction_length, train_ds, epochs, num_layers, batch_size):
        estimator = DeepAREstimator(freq=freq, prediction_length=prediction_length,
                                        trainer=Trainer(ctx="cpu", epochs=epochs, batch_size = batch_size, num_batches_per_epoch = 100), num_layers = num_layers)
        predictor = estimator.train(training_data=train_ds)
        # Save the predictor
        predictor.serialize(Path("P:/0. R&D/6. SKU sales forecast/4_Predictors/"))
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















