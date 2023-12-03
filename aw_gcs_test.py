import mlflow
#from google.cloud import storage
from params import *
from mlflow.tracking import MlflowClient
from google.cloud import storage
import tensorflow as tf
from tensorflow import keras
import os

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
    """Upload a local model directory to GCS.

    Args:
        local_model_path (str): Local path to the model directory.
        bucket_name (str): Name of the GCS bucket.
        destination_blob_name (str): Name to use for the GCS blob.

    Returns:
        None
    """
    # Initialize the GCS client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Create a blob with the desired name
    blob = bucket.blob(destination_blob_name)

    # Upload the model directory to GCS
    blob.upload_from_folder(local_model_path)

    print(f"Model uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")






if __name__ == "__main__":
    #model = load_model_aw(name=f'my_NLP_CNN_MLFLOW_model_{2}')
    #model.save("test_3")
    #upload_model_to_gcs("test_1/saved_model.pb", "ratemate_test_2", "model_2")

    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

    latest_blob = max(blobs, key=lambda x: x.updated)
    latest_model_path_to_save = "test_test"
    latest_blob.download_to_filename(latest_model_path_to_save)
    print("finish")
    latest_model = keras.models.load_model(latest_model_path_to_save)

    print("✅ Latest model downloaded from cloud storage")
