.DEFAULT_GOAL := run_streamlit
run_api:
	uvicorn api.fast:app --reload

run_streamlit:
	streamlit run streamlit/app.py
