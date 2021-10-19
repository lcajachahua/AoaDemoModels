from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradatasqlalchemy.types import VARCHAR, BIGINT, CLOB
from sklearn import metrics
from aoa.sto.util import save_metadata, save_evaluation_metrics

import os
import numpy as np
import json
import base64
import dill


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    def eval_partition(partition):
        target_column = data_conf["target_column"]
        partition = partition.read()
        model_artefact = partition.loc[partition['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        categorical_columns = data_conf["categorical_columns"]
        partition[categorical_columns] = partition[categorical_columns].astype("category")
        X_test = partition[model.features]
        y_test = partition[[target_column]]

        y_pred = model.predict(X_test)

        partition_id = partition.partition_ID.iloc[0]

        # record whatever partition level information you want like rows, data stats, metrics, explainability, etc
        partition_metadata = json.dumps({
            "num_rows": partition.shape[0],
            "metrics": {
                "MAE": "{:.2f}".format(metrics.mean_absolute_error(y_test, y_pred)),
                "MSE": "{:.2f}".format(metrics.mean_squared_error(y_test, y_pred)),
                "R2": "{:.2f}".format(metrics.r2_score(y_test, y_pred))
            }
        })

        return np.array([[partition_id, partition.shape[0], partition_metadata]])
        # we join the model artefact to the 1st row of the data table so we can load it in the partition
    
    partition_id = "center_id" #data_conf["partition_column"]
    query = f"""
        SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.{partition_id} ORDER BY x.{partition_id}) AS n_row 
        FROM {data_conf["data_table"]} x) AS d
        LEFT JOIN aoa_sto_models m
        ON d.{partition_id} = CAST(m.partition_id AS BIGINT)
        WHERE m.model_version = '{model_version}'
    """
    #query
    df = DataFrame(query=query)
    eval_df = df.map_partition(lambda partition: eval_partition(partition),
                               data_partition_column=partition_id,
                               returns=dict([("partition_id", VARCHAR()),
                                             ("num_rows", BIGINT()),
                                             ("partition_metadata", CLOB())]),
                              )

    # materialize as we reuse result
    #eval_df = DataFrame(eval_df._table_name, materialize=True)

    save_metadata(eval_df)
    save_evaluation_metrics(eval_df, ["MAE", "MSE", "R2"])

    print("Finished evaluation")
