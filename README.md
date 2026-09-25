# House Price Prediction API

A FastAPI service that estimates California house values from neighborhood-level
housing features. The model is a scikit-learn `RandomForestRegressor` trained on
the California Housing dataset. Train and validate the model before starting the
API: the service loads its saved model files when `main.py` is imported.

## Project Files

| File | Purpose |
| --- | --- |
| `train.py` | Loads the California Housing dataset, trains and evaluates the model, then saves the model and feature list. |
| `main.py` | Defines the FastAPI app and its health, single-prediction, and CSV batch-prediction endpoints. |
| `main_single_prediction.py` | Separate single-prediction API implementation; the workflow below uses `main.py`. |
| `explore.py` | Prints a preview and descriptive statistics for the dataset. |
| `requirements.txt` | Lists the Python packages used by the project. |

## Requirements

- Python 3.10 or later
- Internet access the first time training runs, so scikit-learn can download the California Housing dataset if it is not cached

## 1. Create an Environment and Install Dependencies

From the project root, create and activate a virtual environment, then install
the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows Command Prompt, activate the environment with
`.venv\Scripts\activate` instead of the `source` command.

## 2. Train and Validate the Model

Run training from the project root:

```bash
python train.py
```

The script fetches the California Housing dataset and separates it into training
and test sets. The test set contains 20% of the records; `random_state=42` makes
the split reproducible. A 100-tree random forest is fitted using only the
training set, then evaluated against the held-out test set.

Training prints these evaluation metrics:

- **Mean Squared Error (MSE):** error in squared target units. The dataset target
	is measured in hundreds of thousands of dollars, so this value is not itself
	a dollar amount.
- **Mean Absolute Error (MAE):** average absolute prediction error, converted to
	dollars for easier interpretation.
- **Root Mean Squared Error (RMSE):** square root of MSE, also converted to
	dollars. RMSE penalizes larger errors more heavily than MAE.
- **R² score:** proportion of target variance explained by the model on the test
	split. A higher value indicates a closer fit on that split; it is not a
	guarantee of accuracy for every future prediction.

Review these test metrics before serving predictions. Successful training also
creates two files in the project root:

- `house_model.joblib` contains the fitted random forest.
- `house_features.joblib` contains the feature names in model order.

Keep both files together and retrain with `train.py` whenever you want to
regenerate the model artifacts. They are required by `main.py` at startup.

## 3. Start the API

With the virtual environment active and both model files present, run:

```bash
python -m uvicorn main:app --reload
```

Run this command from the project root. The development server is available at
`http://127.0.0.1:8000` by default. `--reload` is intended for local development;
use an appropriate production ASGI server configuration when deploying.

Useful endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic welcome response and endpoint information. |
| `GET` | `/health` | Reports API status and the expected model features. |
| `POST` | `/predict` | Predicts a value from one JSON record. |
| `POST` | `/predict-file` | Predicts values for records in an uploaded CSV and returns a CSV download. |
| `GET` | `/docs` | Interactive Swagger UI for trying the endpoints. |

## 4. Make a Single Prediction

Send one JSON object to `/predict`. All eight numeric fields are required. The
latitude and longitude must fall within the API's California bounds; the other
feature values must be greater than zero.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"MedInc": 8.3252,
		"HouseAge": 41,
		"AveRooms": 6.9841,
		"AveBedrms": 1.0238,
		"Population": 322,
		"AveOccup": 2.5556,
		"Latitude": 37.88,
		"Longitude": -122.23
	}'
```

The response includes `predicted_price` as a formatted dollar value,
`prediction_price_short` in the dataset's target units (hundreds of thousands of
dollars), and `confidence_range`. The range is currently a fixed $25,000 below
and above the prediction; it is a convenience estimate, not a statistically
calibrated prediction interval.

## 5. Make Batch Predictions from a CSV

Create a CSV with one property per row and these exact column names:

```csv
MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude
8.3252,41,6.9841,1.0238,322,2.5556,37.88,-122.23
5.6431,25,5.8174,1.0736,1200,2.1098,34.05,-118.25
```

Upload it to `/predict-file` as multipart form data. For example, if the input
file is named `housing_rows.csv`:

```bash
curl -X POST "http://127.0.0.1:8000/predict-file" \
	-F "file=@housing_rows.csv" \
	--output predictions.csv
```

The endpoint requires a `.csv` file containing all eight feature columns. Extra
columns in the input are preserved. The downloaded `predictions.csv` contains
the original columns plus `Predicted_price_usd` (numeric dollars, rounded to two
decimal places) and `Predicted_price_formatted` (a dollar-formatted value).
Invalid or empty CSV files, files with missing required columns, and non-CSV
filenames are rejected with an error response.

## Troubleshooting

- **Model file not found at startup:** run `python train.py` from the project
	root, then start Uvicorn from that same directory.
- **Dataset download fails:** check internet access and rerun `python train.py`.
- **CSV request is rejected:** confirm the filename ends in `.csv`, the file has
	at least one data row, and each required feature column is present with the
	exact spelling and capitalization shown above.
- **Need to inspect or try the API manually:** open `http://127.0.0.1:8000/docs`
	while the server is running.