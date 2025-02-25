import os
import pandas as pd
from datetime import datetime

import batch


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_integration():
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL,
        }
    }

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        'PULocationID',
        'DOLocationID',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
    ]
    df = pd.DataFrame(data, columns=columns)
    print(df)

    output_file = batch.get_output_path(2023, 1)

    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options,
    )

    # os.system("aws s3 ls s3://nyc-duration --recursive")
    os.system('python batch.py 2023 1')

    df_actual = pd.read_parquet(output_file, storage_options=options)
    print("sum of predicted durations:", df_actual['predicted_duration'].sum())

    assert df_actual['predicted_duration'].sum() != 0
