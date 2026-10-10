
# ============================================================
# WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS
# Project: Smart Logistics Performance Analysis Using Python
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# STEP 1: Set file paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "week2_cleaned_logistics_data.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("WEEK 4: LOGISTICS PREDICTIVE MODELING")
print("=" * 60)

# ------------------------------------------------------------
# STEP 2: Load the dataset
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

# ------------------------------------------------------------
# STEP 3: Prepare dates and target variable
# ------------------------------------------------------------

for column in ["Dispatch_Date", "Actual_Delivery"]:
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column], errors="coerce"
        )

if "Dispatch_Date" not in df.columns:
    raise ValueError("Dispatch_Date column is missing.")

if "Actual_Delivery" not in df.columns:
    raise ValueError("Actual_Delivery column is missing.")

# Calculate actual delivery duration.
df["Delivery_Days"] = (
    df["Actual_Delivery"] - df["Dispatch_Date"]
).dt.days

# Keep only shipments with known, non-negative delivery duration.
df = df.dropna(subset=["Delivery_Days"]).copy()
df = df[df["Delivery_Days"] >= 0].copy()

df["Delivery_Days"] = df["Delivery_Days"].astype(float)

print("\nCompleted shipments:", len(df))
print("Pending or unavailable delivery records excluded.")

# ------------------------------------------------------------
# STEP 4: Select prediction features
# ------------------------------------------------------------

candidate_features = [
    "Distance_KM",
    "Shipping_Cost",
    "Weight_KG"
]

features = [
    column for column in candidate_features
    if column in df.columns
]

if not features:
    raise ValueError(
        "No numeric prediction features were found."
    )

if len(df) < 8:
    raise ValueError(
        "Too few completed shipments for this experiment."
    )

X = df[features].copy()

# Ensure input features are numeric.
for column in features:
    X[column] = pd.to_numeric(
        X[column], errors="coerce"
    )

y = df["Delivery_Days"]

print("\nInput features:", features)
print("Target variable: Delivery_Days")

# ------------------------------------------------------------
# STEP 5: Split data into training and testing sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ------------------------------------------------------------
# STEP 6: Build and train the model
# ------------------------------------------------------------

model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# ------------------------------------------------------------
# STEP 7: Predict delivery duration
# ------------------------------------------------------------

predictions = model.predict(X_test)

# Delivery duration cannot be negative.
predictions = np.maximum(predictions, 0)

results = pd.DataFrame({
    "Actual_Delivery_Days": y_test.to_numpy(),
    "Predicted_Delivery_Days": predictions
})

results["Absolute_Error_Days"] = np.abs(
    results["Actual_Delivery_Days"]
    - results["Predicted_Delivery_Days"]
)

# ------------------------------------------------------------
# STEP 8: Evaluate model performance
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test, predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n" + "=" * 45)
print("MODEL PERFORMANCE")
print("=" * 45)
print(f"MAE: {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")
print(f"R-squared: {r2:.3f}")

# Save model metrics.
metrics = pd.DataFrame({
    "Metric": ["MAE_days", "RMSE_days", "R_squared"],
    "Value": [mae, rmse, r2]
})

# ------------------------------------------------------------
# STEP 9: Cross-validation and CSV saving
# ------------------------------------------------------------

print("\n" + "=" * 45)
print("CROSS-VALIDATION")
print("=" * 45)

number_of_folds = min(5, len(X_train))

if number_of_folds >= 2:
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=number_of_folds,
        scoring="neg_mean_absolute_error"
    )

    cv_mae_scores = -cv_scores

    cv_results = pd.DataFrame({
        "Fold": range(1, number_of_folds + 1),
        "MAE_Days": cv_mae_scores
    })

    cv_file = os.path.join(
        OUTPUT_DIR, "week4_cross_validation.csv"
    )

    cv_results.to_csv(cv_file, index=False)

    print("Cross-validation MAE for each fold:")
    print(cv_results.to_string(index=False))
    print(
        f"Average CV MAE: {cv_mae_scores.mean():.2f} days"
    )
    print("Cross-validation CSV:", cv_file)
    print("File exists after saving:", os.path.exists(cv_file))

else:
    print("Not enough training records for cross-validation.")

# ------------------------------------------------------------
# STEP 10: Save prediction results and metrics
# ------------------------------------------------------------

prediction_file = os.path.join(
    OUTPUT_DIR, "week4_prediction_results.csv"
)

metrics_file = os.path.join(
    OUTPUT_DIR, "week4_model_metrics.csv"
)

results.to_csv(prediction_file, index=False)
metrics.to_csv(metrics_file, index=False)

print("\nPrediction results saved:", prediction_file)
print("Metrics saved:", metrics_file)
print("Prediction file exists:", os.path.exists(prediction_file))
print("Metrics file exists:", os.path.exists(metrics_file))

# ------------------------------------------------------------
# STEP 11: Create actual vs predicted graph
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Actual_Delivery_Days"],
    results["Predicted_Delivery_Days"],
    alpha=0.8,
    label="Test shipments"
)

minimum = min(
    results["Actual_Delivery_Days"].min(),
    results["Predicted_Delivery_Days"].min()
)

maximum = max(
    results["Actual_Delivery_Days"].max(),
    results["Predicted_Delivery_Days"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Ideal prediction"
)

plt.xlabel("Actual Delivery Duration (Days)")
plt.ylabel("Predicted Delivery Duration (Days)")
plt.title("Actual vs Predicted Delivery Time")
plt.legend()
plt.tight_layout()

graph_file = os.path.join(
    OUTPUT_DIR, "week4_actual_vs_predicted.png"
)

plt.savefig(graph_file, dpi=300)
plt.close()

print("\nPrediction graph saved:", graph_file)
print("Graph file exists:", os.path.exists(graph_file))

# ------------------------------------------------------------
# STEP 12: Create demonstration optimization recommendations
# ------------------------------------------------------------

# This is a simple planning rule for demonstration only.
# It does not prove that costs or delays have been reduced.

threshold = y_train.median()

recommendations = results.copy()

recommendations["Planning_Recommendation"] = np.where(
    recommendations["Predicted_Delivery_Days"] > threshold,
    "Review route and dispatch priority",
    "Standard monitoring"
)

recommendation_file = os.path.join(
    OUTPUT_DIR, "week4_optimization_recommendations.csv"
)

recommendations.to_csv(
    recommendation_file, index=False
)

print("\nOptimization recommendations saved:")
print(recommendation_file)
print(
    "Recommendation file exists:",
    os.path.exists(recommendation_file)
)

# ------------------------------------------------------------
# STEP 13: Final verification
# ------------------------------------------------------------

expected_files = [
    "week4_prediction_results.csv",
    "week4_model_metrics.csv",
    "week4_cross_validation.csv",
    "week4_actual_vs_predicted.png",
    "week4_optimization_recommendations.csv"
]

print("\n" + "=" * 45)
print("OUTPUT FILE VERIFICATION")
print("=" * 45)

for filename in expected_files:
    full_path = os.path.join(OUTPUT_DIR, filename)
    print(
        f"{filename}: "
        f"{'CREATED' if os.path.exists(full_path) else 'NOT CREATED'}"
    )

print("\nWeek 4 program finished.")
print("Output folder:", OUTPUT_DIR)
