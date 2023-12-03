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

def upload_model_to_gcs_0(local_model_path, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_model_path)

    # if MODEL_TARGET == "gcs":
    #     # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

    #     model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    #     client = storage.Client()
    #     bucket = client.bucket(BUCKET_NAME)
    #     blob = bucket.blob(f"models/{model_filename}")
    #     blob.upload_from_filename(model_path)

    #     print("✅ Model saved to GCS")

        # return None

def load_saved_model_from_gcs(gcs_uri):
    try:
        # Load the TensorFlow SavedModel directly from GCS
        loaded_model = tf.saved_model.load(gcs_uri)
        print("TensorFlow SavedModel loaded successfully.")
        return loaded_model
    except Exception as e:
        print(f"Failed to load the TensorFlow SavedModel. Error: {e}")
        return None



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

    # Iterate over files in the local directory and upload each one
    for root, dirs, files in os.walk(local_model_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            blob.upload_from_filename(local_file_path, destination=os.path.join(destination_blob_name, file))

    print(f"Model uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")









if __name__ == "__main__":
    #model = load_model_aw(name=f'my_NLP_CNN_MLFLOW_model_{2}')
    #model.save("test_3")
    #upload_model_to_gcs("test_1/saved_model.pb", "ratemate_test_2", "model_2")

    # Example usage
    #gcs_uri = 'gs://ratemate_test_2/model_2'
    #loaded_saved_model = load_saved_model_from_gcs(gcs_uri)
# Example usage
    local_model_path = "test_1"
    bucket_name = "ratemate_test_2"
    destination_blob_name = "model_3"

    upload_model_to_gcs(local_model_path, bucket_name, destination_blob_name)
