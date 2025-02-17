if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
    
import mlflow
import pickle

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment('homework_03')

@data_exporter
def export_data_to_mlflow(data, *args, **kwargs):

    dv, lr = data

    with mlflow.start_run():

        mlflow.sklearn.autolog()

        mlflow.sklearn.log_model(lr, "linear_regression_model")
        print("Linear Regression model logged to MLflow.")


        with open('mlflow_data/dict_vectorizer.bin', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("mlflow_data/dict_vectorizer.bin")
        print("DictVectorizer artifact saved and logged to MLflow.")

    print('...Done')