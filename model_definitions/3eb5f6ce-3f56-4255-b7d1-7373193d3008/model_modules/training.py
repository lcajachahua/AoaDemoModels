from teradataml import DataFrame, create_context, remove_context, copy_to_sql
from teradatasqlalchemy.types import VARCHAR, BIGINT, CLOB
from aoa.sto.util import save_metadata, cleanup_cli

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

import os
import json
import base64
import dill
import uuid


def train(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    cleanup_cli(model_version)

    def train_partition(partition, model_version, hyperparams):
        partition = partition.read() # read returns pandas df
        numeric_columns = data_conf["numeric_columns"]
        target_column = data_conf["target_column"]
        categorical_columns = data_conf["categorical_columns"]
        features = numeric_columns + categorical_columns

        # modelling pipeline: feature encoding + algorithm
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))])
        oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        regressor = RandomForestRegressor(random_state=hyperparams["rand_seed"],
                                          n_estimators=hyperparams["n_estimators"]
                                         )
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", oh_encoder, categorical_columns)])

        model = Pipeline([("preprocessor", preprocessor),
                             ("regressor", regressor)])

        # data loading and formating
        train_df = partition[features + [target_column]]
        train_df[categorical_columns] = train_df[categorical_columns].astype("category")

        print('Loaded data ...')
        # preprocess training data and train the model
        X_train = train_df[features]
        y_train = train_df[target_column]
        model.fit(X_train, y_train)

        print("Finished training")
        model.features = features

        partition_id = partition.loc[0, 'center_id']
        artefact = base64.b64encode(dill.dumps(model))

        # record whatever partition level information you want like rows, 
        # data stats, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": partition.shape[0],
            "hyper_parameters": hyperparams
        })
        return np.array([partition_id, model_version, partition.shape[0], partition_metadata, artefact])

    print("Starting training...")

    df = DataFrame.from_query("SELECT * FROM {table}".format(table=data_conf["data_table"]))
    model_df = df.map_partition(lambda partition: train_partition(partition, model_version, hyperparams),
                                data_partition_column="center_id",
                                returns=dict([("partition_id", VARCHAR()),
                                          ("model_version", VARCHAR()),
                                          ("num_rows", BIGINT()),
                                          ("partition_metadata", CLOB()),
                                          ("model_artefact", CLOB())])#,
                                )
    # materialize as we reuse result
    #model_df = DataFrame(model_df._table_name, materialize=True)

    # append to models table
    #model_df.to_sql("aoa_sto_models", if_exists="append")
    copy_to_sql(df=model_df, table_name="aoa_sto_models", schema_name='AOA_DEMO', if_exists="append", index=True, index_label='index')

    save_metadata(model_df)

    print("Finished training")
    
    remove_context()
