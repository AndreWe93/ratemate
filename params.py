import os
import numpy as np

##################  VARIABLES  ##################
#DATA_SIZE = os.environ.get("DATA_SIZE")
#CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_KERAS_MODEL_NAME = os.environ.get("MLFLOW_KERAS_MODEL_NAME")
MLFLOW_SKLEARN_MODEL_NAME = os.environ.get("MLFLOW_SKLEARN_MODEL_NAME")
#MLFLOW_RUN_ID = os.environ.get("MLFLOW_RUN_ID")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), 'code', "ratemate", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), 'code', "ratemate", "mlops", "training_outputs")

# COLUMN_NAMES_RAW = [
#         "placeId",
#         "title",
#         "reviewId",
#         "reviewerId",
#         "isLocalGuide",
#         "reviewDetailedRating/Atmosphere",
#         "reviewDetailedRating/Food",
#         "reviewDetailedRating/Service",
#         "reviewerNumberOfReviews",
#         "text",
#         "textTranslated",
#         "stars"
#     ]

COLUMN_NAMES_RAW = ['title', 'reviewId', 'reviewDetailedRating/Food', 'reviewDetailedRating/Service', 'reviewDetailedRating/Atmosphere', 'text', 'textTranslated', 'stars']

DTYPES_RAW = {}

DTYPES_PROCESSED = np.float32



################## VALIDATIONS #################

env_valid_options = dict(
    #DATA_SIZE=["1k", "20k", "all"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")

for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
