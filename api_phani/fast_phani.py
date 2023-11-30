import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#app.state.model = load_model()

@app.get("/predict")
def predict(
    url: str,
    price_review_weightage: float = Query(..., ge=0, le=1),
    food_review_weightage: float = Query(..., ge=0, le=1),
    service_review_weightage: float = Query(..., ge=0, le=1),
    ambience_review_weightage: float = Query(..., ge=0, le=1),
    local_guides_review_weightage: bool = Query(...),
):    # 1
    """
    """
    # model = app.state.model
    # assert model is not None
    # X_pred = pd.DataFrame(locals(), index=[0])
    # # Convert to US/Eastern TZ-aware!
    # X_pred['pickup_datetime'] = pd.Timestamp(pickup_datetime, tz='US/Eastern')
    # X_processed = preprocess_features(X_pred)
    # y_pred = model.predict(X_processed)
    # return dict(fare_amount=float(y_pred))

@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}

# import pandas as pd
# from fastapi import FastAPI, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse

# app = FastAPI()

# # Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Use Jinja2 for HTML templates
# templates = Jinja2Templates(directory="templates")


# @app.get("/predict_result", response_class=HTMLResponse)
# def predict_result(
#     request: HTMLResponse,
#     url: str,
#     price_review_weightage: float = Query(..., ge=0, le=1),
#     food_review_weightage: float = Query(..., ge=0, le=1),
#     service_review_weightage: float = Query(..., ge=0, le=1),
#     ambience_review_weightage: float = Query(..., ge=0, le=1),
# ):
#     # Rest of the function remains unchanged
#     # ...

#     # Example: Access the model from the app state (uncomment after loading the model)
#     # model = app.state.model
#     # assert model is not None

#     # Example: Create a DataFrame from input parameters
#     input_data = pd.DataFrame(
#         {
#             'url': [url],
#             'price_review_weightage': [price_review_weightage],
#             'food_review_weightage': [food_review_weightage],
#             'service_review_weightage': [service_review_weightage],
#             'ambience_review_weightage': [ambience_review_weightage],
#         }
#     )

#     # Example: Perform prediction using the loaded model
#     # y_pred = model.predict(input_data)
#     # Replace the following line with your actual prediction logic
#     prediction_result = {"prediction": "Replace this with your actual prediction result"}

#     return templates.TemplateResponse("predict_result.html", {"request": request, "result": prediction_result})
