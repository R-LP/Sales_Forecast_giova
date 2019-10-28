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


