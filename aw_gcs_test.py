import mlflow
#from google.cloud import storage
from params import *
from mlflow.tracking import MlflowClient
from google.cloud import storage


def load_model_aw(name, stage="Production"):
    print(f"\nLoad [{stage}] model from MLflow..." )

    # Load model from MLflow
    model = None
    # $CHA_BEGIN
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        model_versions = client.get_latest_versions(name=name, stages=[stage])
        model_uri = model_versions[0].source

        assert model_uri is not None
    except:
        print(f"\n❌ No model found with name {name} in stage {stage}")

        return None

    model = mlflow.tensorflow.load_model(model_uri=model_uri)

    print("✅ Model loaded from MLflow")
    # $CHA_END
    return model

def upload_model_to_gcs(local_model_path, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_model_path)



if __name__ == "__main__":
    model = load_model_aw(name=f'my_NLP_CNN_MLFLOW_model_{2}')
    model.save("test_3")
    upload_model_to_gcs("test_2/saved_model.pb", "ratemate_test", "model_2")
