.DEFAULT_GOAL := run_streamlit_copy_old
run_api:
	uvicorn api.fast:app --reload

run_streamlit_copy_old:
	streamlit run streamlit/app_copy_old.py

run_streamlit:
	streamlit run streamlit/app.py
