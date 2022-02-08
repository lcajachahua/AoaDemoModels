from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame

import joblib
import os


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"])

    feature_names = ["EDUCATION_1","SEX","PAY_1","AGE","LIMIT_BAL","SUM_LPAY_REC","STD_LBILL_TOT","CV_LPAY_TOT","CV_LBILL_TOT","STD_LPAY_TOT","CANT_PAY_MAY0",
                     "BILL_AMT1","RATE_PAY_BILL1","LOG_BILL_AMT1","SUM_LBILL_REC","AVG_LBILL_TOT","AVG_LPAY_TOT","STD_PAY_TOT"]
    target_name = "TAR"

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame(data_conf["table"])
    train_df = train_df.select([feature_names + [target_name]])
    train_df = train_df.to_pandas()

    # split data into X and y
    X_train = train_df.drop(target_name, 1)
    y_train = train_df[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                     ('xgb', XGBClassifier(eta=hyperparams["eta"],
                                           max_depth=hyperparams["max_depth"], n_estimators=hyperparams["n_estimators"]))])
    # xgboost saves feature names but lets store on pipeline for easy access later
    model.feature_names = feature_names
    model.target_name = target_name

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/xgb_model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name, pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")