.PHONY: install data sample app api test eval docker

install:
	pip install -r requirements.txt

data:        ## real Property Price Register data
	python -m src.pipeline

sample:      ## synthetic data for offline testing
	python -m src.pipeline --sample

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.main:app --reload

test:
	pytest -q

eval:        ## benchmark the LLM text-to-SQL layer
	python -m src.llm.evaluate

docker:
	docker build -t irish-housing . && docker run -p 8501:8501 irish-housing
