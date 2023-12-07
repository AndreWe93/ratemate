import glob
import os
import time
import pickle

from tensorflow import keras

from params import *
import mlflow
import sklearn
from mlflow.tracking import MlflowClient




columns = [#"reviewContext/Price per person",
        #"reviewContext/Service",
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service",
        'reviews_without_SW',
        'reviews_with_SW',
        'stars'
        ]
y_columns = [#"reviewContext/Price per person",
        #"reviewContext/Service",
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service"
        ]
X_column = ['reviews_without_SW',
        #'reviews_with_SW'
        ]

numeric_columns = ['stars']

new_columns_names = [#"price_rating",
                     "atmosphere_rating",
                     "food_rating",
                     "service_rating"
                     ]





def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    if MODEL_TARGET == "mlflow":
        # $CHA_BEGIN
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on MLflow")
        # $CHA_END

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # # Save params locally
    if MODEL_TARGET == "local":

        if params is not None:
            params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
            with open(params_path, "wb") as file:
                pickle.dump(params, file)

        # Save metrics locally
        if metrics is not None:
            metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
            with open(metrics_path, "wb") as file:
                pickle.dump(metrics, file)

        print("✅ Results saved locally")


def save_model(keras_model: keras.Model = None, sklearn_model: sklearn.base.BaseEstimator = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if MODEL_TARGET == "local":
        # Save model locally
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
        keras_model.save(model_path)

        print(f"✅ Model saved locally: {model_path}")

    # if MODEL_TARGET == "gcs":
    #     # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

    #     model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    #     client = storage.Client()
    #     bucket = client.bucket(BUCKET_NAME)
    #     blob = bucket.blob(f"models/{model_filename}")
    #     blob.upload_from_filename(model_path)

    #     print("✅ Model saved to GCS")

        # return None

    if MODEL_TARGET == "mlflow" and keras_model is not None:

        # $CHA_BEGIN
        mlflow.tensorflow.log_model(
            model=keras_model,
            artifact_path="model",
            registered_model_name=MLFLOW_KERAS_MODEL_NAME
        )

        print("✅ Model saved to MLflow")

        return None
        # $CHA_END

    if MODEL_TARGET == "mlflow" and sklearn_model is not None:
        mlflow.sklearn.log_model(
            sk_model=sklearn_model,
            artifact_path="model",
            registered_model_name=MLFLOW_SKLEARN_MODEL_NAME
        )

        print("✅ Model saved to MLflow")

        return None

    return None


def load_model(name,model_type, stage="Production",):
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found
    model_type = RF/ TF/ CB
    """

    if MODEL_TARGET == "local":
        print(f"\nLoad latest model from local registry..." )

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(f"\nLoad latest model from disk...")
        if model_type == 'TF':
            latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        elif model_type == 'RF' or model_type == 'CB':
            latest_model = sklearn.load_model(most_recent_model_path_on_disk)


        print("✅ Model loaded from local disk")

        return latest_model

    #elif MODEL_TARGET == "gcs":
    #     # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
    #     print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

    #     client = storage.Client()
    #     blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

    #     try:
    #         latest_blob = max(blobs, key=lambda x: x.updated)
    #         latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
    #         latest_blob.download_to_filename(latest_model_path_to_save)

    #         latest_model = keras.models.load_model(latest_model_path_to_save)

    #         print("✅ Latest model downloaded from cloud storage")

    #         return latest_model
    #     except:
    #         print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

    #        return None

    elif MODEL_TARGET == "mlflow":
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
        if model_type == 'TF':
            model = mlflow.tensorflow.load_model(model_uri=model_uri)

            print("✅ Model loaded from MLflow")
        elif model_type == 'RF' or model_type == 'CB':
            model = mlflow.sklearn.load_model(model_uri=model_uri)

            print("✅ Model loaded from MLflow")
        # $CHA_END
        return model

    else:
        return None




def mlflow_transition_model(name: str,current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=name, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {name} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=name,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {name} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper





def check_models(your_experiment_name):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(your_experiment_name)
    runs = mlflow.search_runs()
    runs_artifacts = runs.artifact_uri
    for i in range(len(runs_artifacts)):
        print(runs_artifacts[i])
