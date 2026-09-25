from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import joblib

print("Loading California housing dataset...")
data = fetch_california_housing()
x=pd.DataFrame(data.data, columns=data.feature_names)
y=data.target

print(f"total records: {x.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = model.score(X_test, y_test)
print(f"Mean Squared Error (target units squared): {mse:.4f}")
print(f"Mean Absolute Error: ${mae * 100000:,.2f}")
print(f"Root Mean Squared Error: ${rmse * 100000:,.2f}")
print(f"R^2 Score: {r2}")

joblib.dump(model, "house_model.joblib")
joblib.dump(list(x.columns),"house_features.joblib")