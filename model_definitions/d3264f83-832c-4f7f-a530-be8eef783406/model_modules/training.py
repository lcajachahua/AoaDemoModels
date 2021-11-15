from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame

import pandas as pd
import numpy as np
import joblib
import os


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]
    
    xls = pd.ExcelFile('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls')
    df = xls.parse('Data', skiprows=1, index_col=None, na_values=['NA'])
    df = df.rename(columns={'default payment next month': 'DEFAULT'})
        
    # Convertimos SEX en dummy
    df.SEX=df.SEX-1

    # Creamos tres Variables Dummy para EDUCATION
    df['EDUCATION_1']=[1 if i == 1 else 0 for i in df['EDUCATION']]
    df['EDUCATION_2']=[1 if i == 2 else 0 for i in df['EDUCATION']]
    df['EDUCATION_3']=[1 if i == 3 else 0 for i in df['EDUCATION']]

    # Creamos dos Variables Dummy para MARRIAGE
    df['MARRIAGE_1']=[1 if i == 1 else 0 for i in df['MARRIAGE']]
    df['MARRIAGE_2']=[1 if i == 2 else 0 for i in df['MARRIAGE']]

    ## Generar variables Cuantitativas transformadas
    LIST_BILL = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    for i in LIST_BILL: 
        df.loc[df.loc[:,i]==-1,i]=0
    
    df['LOG_BILL_AMT1'] = round(np.log1p(df['BILL_AMT1']),5)
    df['LOG_BILL_AMT2'] = round(np.log1p(df['BILL_AMT2']),5)
    df['LOG_BILL_AMT3'] = round(np.log1p(df['BILL_AMT3']),5)
    df['LOG_BILL_AMT4'] = round(np.log1p(df['BILL_AMT4']),5)
    df['LOG_BILL_AMT5'] = round(np.log1p(df['BILL_AMT5']),5)
    df['LOG_BILL_AMT6'] = round(np.log1p(df['BILL_AMT6']),5)
    df['LOG_PAY_AMT1'] = round(np.log1p(df['PAY_AMT1']),5)
    df['LOG_PAY_AMT2'] = round(np.log1p(df['PAY_AMT2']),5)
    df['LOG_PAY_AMT3'] = round(np.log1p(df['PAY_AMT3']),5)
    df['LOG_PAY_AMT4'] = round(np.log1p(df['PAY_AMT4']),5)
    df['LOG_PAY_AMT5'] = round(np.log1p(df['PAY_AMT5']),5)
    df['LOG_PAY_AMT6'] = round(np.log1p(df['PAY_AMT6']),5)
    
    LIST_PAY  = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    LIST_BILL = ['LOG_BILL_AMT1','LOG_BILL_AMT2','LOG_BILL_AMT3','LOG_BILL_AMT4','LOG_BILL_AMT5','LOG_BILL_AMT6']
    LIST_PAMT = ['LOG_PAY_AMT1','LOG_PAY_AMT2','LOG_PAY_AMT3','LOG_PAY_AMT4','LOG_PAY_AMT5','LOG_PAY_AMT6']
    
    # Imputar los valores faltantes con cero
    for i in LIST_BILL: 
        df.loc[df.loc[:,i].isnull(),i]=0
    
    ## Creamos las variables para el entrenamiento o train
    df['SUM_PAY_TOT']    = df[LIST_PAY].sum(axis=1)
    df['STD_PAY_TOT']    = df[LIST_PAY].std(axis=1)
    df['SUM_PAY_REC']    = df['PAY_0'] + df['PAY_2']
    df['CANT_PAY_MAY0']  = df[LIST_PAY].gt(0).sum(axis=1)
    df['AVG_LBILL_TOT']  = df[LIST_BILL].mean(axis=1)
    df['STD_LBILL_TOT']  = df[LIST_BILL].std(axis=1)
    df['CV_LBILL_TOT']   =  df['STD_LBILL_TOT']/(df['AVG_LBILL_TOT']+1)
    df['SUM_LBILL_REC']  = df['LOG_BILL_AMT1'] + df['LOG_BILL_AMT2']
    df['CANT_LBILL_MAY0']= df[LIST_BILL].gt(0).sum(axis=1)
    df['AVG_LPAY_TOT']   = df[LIST_PAMT].mean(axis=1)
    df['STD_LPAY_TOT']   = df[LIST_PAMT].std(axis=1)
    df['CV_LPAY_TOT']    =  df['STD_LPAY_TOT']/(df['AVG_LPAY_TOT']+1)
    df['SUM_LPAY_REC']   = df['LOG_PAY_AMT1'] + df['LOG_PAY_AMT2']
    df['CANT_LPAY_MAY0'] = df[LIST_PAMT].gt(0).sum(axis=1)
    df['RATE_PAY_BILL1'] = df['PAY_AMT1']/(df['BILL_AMT1']+1)
    df['RATE_PAY_BILL2'] = df['PAY_AMT2']/(df['BILL_AMT2']+1)
    
    feature_names = ['EDUCATION_1','SEX','PAY_0','AGE','LIMIT_BAL','SUM_LPAY_REC','STD_LBILL_TOT','CV_LPAY_TOT','CV_LBILL_TOT','STD_LPAY_TOT','CANT_PAY_MAY0','BILL_AMT1','RATE_PAY_BILL1','LOG_BILL_AMT1','SUM_LBILL_REC','AVG_LBILL_TOT','AVG_LPAY_TOT','STD_PAY_TOT']
    target_name = 'DEFAULT'
    
    X_train = df[feature_names]
    y_train = df[target_name]
    
    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                      ('xgb', XGBClassifier(eta=hyperparams["eta"], max_depth=hyperparams["max_depth"], n_estimators=hyperparams["n_estimators"]))])
    
    # xgboost saves feature names but lets store on pipeline for easy access later
    model.feature_names = feature_names
    model.target_name = target_name

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/xgb_model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name, pmml_f_name="artifacts/output/xgb_model.pmml")

    print("Saved trained model")
