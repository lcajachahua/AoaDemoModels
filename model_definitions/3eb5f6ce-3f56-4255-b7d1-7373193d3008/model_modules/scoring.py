from teradataml import create_context
from tdextensions.distributed import DistDataFrame, DistMode

import os
import base64
import dill


def score(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    def score_partition(partition):
        partition = partition.read()
        model_artefact = partition.loc[partition['n_row'] == 1, 'model_artefact'].iloc[0]
        model = dill.loads(base64.b64decode(model_artefact))

        categorical_columns = data_conf["categorical_columns"]
        partition[categorical_columns] = partition[categorical_columns].astype("category")
        X = partition[model.features]

        return model.predict(X)

    # we join the model artefact to the 1st row of the data table so we can load it in the partition
    partition_id = data_conf["partition_column"]
    query = f"""
        SELECT d.*, CASE WHEN n_row=1 THEN m.model_artefact ELSE null END AS model_artefact 
        FROM (SELECT x.*, ROW_NUMBER() OVER (PARTITION BY x.{partition_id} ORDER BY x.{partition_id}) AS n_row 
        FROM {data_conf["data_table"]} x) AS d
        LEFT JOIN aoa_sto_models m
        ON d.{partition_id} = CAST(m.partition_id AS BIGINT)
        WHERE m.model_version = '{model_version}'
    """
    df = DataFrame(query=query)
    scored_df = df.map_partition(lambda partition: score_partition(partition),
                                 data_partition_column=partition_id,
                                 returns=dict([("prediction", VARCHAR())]),
                                )

    scored_df.to_sql(data_conf["results_table"], if_exists="append")