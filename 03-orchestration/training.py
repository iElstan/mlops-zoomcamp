from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def train(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')  # Исправлен отступ

    dv = DictVectorizer()

    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = df[target].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"Intercept: {lr.intercept_}")

    return dv, lr