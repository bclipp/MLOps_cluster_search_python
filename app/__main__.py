import math as math
import sys
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.model_selection import GridSearchCV
from hyperopt import SparkTrials
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def main():
    spark = SparkSession \
        .builder \
        .appName("MLops_search_python") \
        .getOrCreate()
    uid = "testing123" if sys.argv[1] == "" else sys.argv[1]
    spark.conf.set("spark.sql.execution.arrow.enabled", True)
    print(f"reading delta table: dbfs:/datalake/stocks_{uid}/data")
    try:
        df = spark.read.format("delta").load(f"dbfs:/datalake/stocks_{uid}/data")
    except Exception as e:
        print(f"There was an error loading the delta stock table, : error:{e}")
    pdf = df.select("*").toPandas()
    df_2 = pdf.loc[:, ["AdjClose", "Volume"]]
    df_2["High_Low_Pert"] = (pdf["High"] - pdf["Low"]) / pdf["Close"] * 100.0
    df_2["Pert_change"] = (pdf["Close"] - pdf["Open"]) / pdf["Open"] * 100.0
    df_2.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df_2)))
    forecast_col = "AdjClose"
    df_2['label'] = df_2[forecast_col].shift(-forecast_out)
    X = np.array(df_2.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_2['label'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("creating MLflow project")
    mlflow.set_experiment(f"/Users/bclipp770@yandex.com/datalake/stocks/experiments/cluster_{uid}")
    experiment = mlflow.get_experiment_by_name(f"/Users/bclipp770@yandex.com/datalake/stocks/experiments/{uid}")
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("building our model")

    def objective(hypers):
        regr = RandomForestRegressor(max_depth=hypers["max_depth"],
                                     max_features=hypers["max_features"],
                                     min_samples_leaf=hypers["min_samples_leaf"],
                                     min_samples_split=hypers["min_samples_split"],
                                     n_estimators=hypers["n_estimators"]
                                     )
        accuracy = cross_val_score(regr, X_train, y_train).mean()
        return {'loss': -accuracy, 'status': STATUS_OK}

    search_space = {
        'max_depth': hp.choice('max_depth', range(1, 110)),
        'max_features': hp.choice('max_features', np.arange(0.1, 1.0, 0.1)),
        "min_samples_leaf": hp.choice('min_samples_leaf', range(3, 5)),
        "min_samples_split": hp.choice("min_samples_split", range(8, 12)),
        'n_estimators': hp.choice('n_estimators', range(100, 500))}

    spark_trials = SparkTrials()
    with mlflow.start_run():
        argmin = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=16,
            trials=spark_trials)


if __name__ == "__main__":
    main()
