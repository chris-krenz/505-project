## About

This project is for BU CS505: NLP.  It is a Logistic Regression classifier that takes in a natural language description of a synthetic biology workflow and produces the specific labotory services that correspond to that description.  Please see the accompanying report in the doc folder for additional information.

## Install

### Dependencies

Create a virtual environment and install dependencies (note: for Win, instead use venv\Scripts\activate.bat to activate the venv):

```console
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

If you want to just use the existing data and models, you can just run the benchmarker.

```console
python src/benchmarker.py
```

If you want to run all steps, including data fetching and processing, you can run the following scripts first:

```console
python src/fetcher.py
python src/extractor.py
python src/preprocessor.py
python src/labeler.py
```

If you want to run the model on a specific sentence, you can use the classify_sentence.py script.

```console
python src/classify_sentence.py --model data/logistic_regression_model.pkl --vectorizer data/tfidf_vectorizer.pkl --keywords src/KEYWORD_LABEL_MAP.json
```

## Contributors

Chris Krenz

## License

[MIT License](LICENSE)
