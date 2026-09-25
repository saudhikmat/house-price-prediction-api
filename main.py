import io
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices based on California housing dataset features.",
    version="1.0.0",
)


model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


# input schema for the API
class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of neighborhood")
    HouseAge: float = Field(gt=0, description="Avarage age in block group")
    AveRooms: float = Field(gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(
        gt=0, description="Average number of bedrooms per household"
    )
    Population: float = Field(gt=0, description="Block group population")
    AveOccup: float = Field(gt=0, description="Average number of household members")
    Latitude: float = Field(gt=32, le=42, description="Block group latitude")
    Longitude: float = Field(gt=-125, le=-114, description="Block group longitude")


@app.get("/")
def home():
    return {
        "message": "Welcome to the House Price Prediction API. Use the /predict endpoint to get predictions.",
        "status": "API is running",
        "endpoints": {
            "/predict": "POST endpoint to predict house prices based on input features.",
            "/docs": "Interactive API documentation (Swagger UI).",
        },
    }


@app.get("/health")
def health():
    return{
        "status": "API is healthy and running",
        "model": "Random Forest Regressor trained on California",
        "features": features,
        "avg_error": "$25000"
    }

#prediction
@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        input_data=pd.DataFrame(
            [{
                "MedInc": features.MedInc,
                "HouseAge": features.HouseAge,
                "AveRooms": features.AveRooms,
                "AveBedrms": features.AveBedrms,
                "Population": features.Population,
                "AveOccup": features.AveOccup,
                "Latitude": features.Latitude,
                "Longitude": features.Longitude,
            }]
        )

        prediction = model.predict(input_data)[0]
        price_prediction = round(prediction * 100000, 2)  # Convert to dollars and round to 2 decimal places
        return {
            "predicted_price": f"${price_prediction:,.0f}",
            "prediction_price_short": f"${prediction:,.2f} hundreds of thousands",
            "confidence_range": f"${price_prediction - 25000:,.0f} - ${price_prediction + 25000:,.0f}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    contents=await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}") from e

    required_columns=[
        'MedInc',
        'HouseAge',
        'AveRooms',
        'AveBedrms',
        'Population',
        'AveOccup',
        'Latitude',
        'Longitude'
    ]

    missing_columns=[col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing_columns)}"
        )


    if len(df)==0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty."
        )


    try:
        predictions=model.predict(df[required_columns])
        df["Predicted_price_usd"] = (predictions * 100000).round(2)
        df["Predicted_price_formatted"] = df["Predicted_price_usd"].apply(
            lambda price: f"${price:,.0f}"
        )

        output=df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions.csv"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
