import pickle
import pandas as pd
import sys
import os

MODEL_FILE = os.getenv('MODEL_FILE', 'model.bin')

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

year = int(sys.argv[1]) # 2023
month = int(sys.argv[2]) # 2
input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'./outputs/output_{year:04d}-{month:02d}.parquet'

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

float(y_pred.std())

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predict'] = y_pred

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

print(f"Mean predicted duration: {y_pred.mean():.2f}")
