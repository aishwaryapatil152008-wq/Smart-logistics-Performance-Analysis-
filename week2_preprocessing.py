
# ============================================================
# PROJECT: Smart Logistics Performance Analysis
# WEEK 2: Data Collection, Cleaning, and Preprocessing
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# STEP 1: Define file paths
# ------------------------------------------------------------

DATA_FILE = os.path.join("data", "logistics_data.csv")
OUTPUT_DIR = "outputs"

# Create outputs folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# STEP 2: Load the dataset
# ------------------------------------------------------------

print("=" * 60)
print("WEEK 2: LOGISTICS DATA PREPROCESSING")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Save original column names for reference
print("\n1. DATA COLLECTION")
print("Dataset loaded successfully.")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# ------------------------------------------------------------
# STEP 3: Explore the dataset
# ------------------------------------------------------------

print("\n2. DATASET PREVIEW")
print(df.head())

print("\n3. COLUMN NAMES")
print(df.columns.tolist())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. DATASET INFORMATION")
df.info()

print("\n6. SUMMARY STATISTICS")
print(df.describe(include="all").to_string())

# ------------------------------------------------------------
# STEP 4: Check missing values
# ------------------------------------------------------------

print("\n7. MISSING VALUE ANALYSIS")

missing_before = df.isnull().sum()
print(missing_before)

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": [int(df[col].isnull().sum()) for col in df.columns],
    "Missing_Percentage": [
        round(df[col].isnull().mean() * 100, 2)
        for col in df.columns
    ]
})

# ------------------------------------------------------------
# STEP 5: Check duplicate records
# ------------------------------------------------------------

print("\n8. DUPLICATE ANALYSIS")

duplicate_count = int(df.duplicated().sum())
print("Exact duplicate rows:", duplicate_count)

# Remove exact duplicate rows from the working copy
cleaned_df = df.drop_duplicates().copy()

print("Rows after removing exact duplicates:", len(cleaned_df))

# ------------------------------------------------------------
# STEP 6: Clean text columns
# ------------------------------------------------------------

print("\n9. TEXT CLEANING")

text_columns = cleaned_df.select_dtypes(
    include=["object", "string"]
).columns

for col in text_columns:
    cleaned_df[col] = cleaned_df[col].apply(
        lambda value: value.strip()
        if isinstance(value, str) else value
    )

# ------------------------------------------------------------
# STEP 7: Convert date columns into datetime format
# ------------------------------------------------------------

print("\n10. DATE CLEANING")

date_columns = [
    "Order_Date",
    "Dispatch_Date",
    "Expected_Delivery",
    "Actual_Delivery"
]

for col in date_columns:
    if col in cleaned_df.columns:
        cleaned_df[col] = pd.to_datetime(
            cleaned_df[col],
            errors="coerce"
        )

print("Date columns converted where available.")

# ------------------------------------------------------------
# STEP 8: Convert numeric columns into numeric format
# ------------------------------------------------------------

print("\n11. NUMERIC DATA CLEANING")

numeric_columns = [
    "Distance_KM",
    "Shipping_Cost",
    "Weight_KG"
]

for col in numeric_columns:
    if col in cleaned_df.columns:
        cleaned_df[col] = pd.to_numeric(
            cleaned_df[col],
            errors="coerce"
        )

# ------------------------------------------------------------
# STEP 9: Handle missing values carefully
# ------------------------------------------------------------

print("\n12. MISSING VALUE HANDLING")

print("Missing values before cleaning:")
print(cleaned_df.isnull().sum())

# Do not invent missing actual delivery dates.
# Preserve them as missing because delivery may be incomplete
# or the information may not have been recorded.

# Fill missing numeric values with the median only for this
# optional analysis copy. The original columns are preserved
# in the main cleaned dataset.
#
# We will NOT fill missing delivery dates or shipment IDs.

print(
    "\nMissing delivery dates are retained as missing. "
    "They are not replaced with invented dates."
)

# ------------------------------------------------------------
# STEP 10: Validate logistics values and dates
# ------------------------------------------------------------

print("\n13. DATA VALIDATION")

validation_issues = []

# Negative distance, cost, or weight is invalid
for col in numeric_columns:
    if col in cleaned_df.columns:
        invalid_count = int((cleaned_df[col] < 0).sum())
        validation_issues.append({
            "Issue": f"Negative values in {col}",
            "Count": invalid_count
        })

# Check whether dispatch date is before order date
if {"Order_Date", "Dispatch_Date"}.issubset(cleaned_df.columns):
    invalid_dispatch = (
        cleaned_df["Dispatch_Date"].notna()
        & cleaned_df["Order_Date"].notna()
        & (cleaned_df["Dispatch_Date"] < cleaned_df["Order_Date"])
    )
    validation_issues.append({
        "Issue": "Dispatch date before order date",
        "Count": int(invalid_dispatch.sum())
    })

# Check actual delivery before order date
if {"Order_Date", "Actual_Delivery"}.issubset(cleaned_df.columns):
    invalid_delivery = (
        cleaned_df["Actual_Delivery"].notna()
        & cleaned_df["Order_Date"].notna()
        & (cleaned_df["Actual_Delivery"] < cleaned_df["Order_Date"])
    )
    validation_issues.append({
        "Issue": "Actual delivery before order date",
        "Count": int(invalid_delivery.sum())
    })

validation_report = pd.DataFrame(validation_issues)
print(validation_report.to_string(index=False))

# ------------------------------------------------------------
# STEP 11: Detect outliers using the IQR method
# ------------------------------------------------------------

print("\n14. OUTLIER DETECTION")

outlier_results = []

for col in numeric_columns:
    if col not in cleaned_df.columns:
        continue

    values = cleaned_df[col].dropna()

    if len(values) < 4:
        outlier_results.append({
            "Column": col,
            "Lower_Bound": np.nan,
            "Upper_Bound": np.nan,
            "Outlier_Count": 0
        })
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (cleaned_df[col] < lower_bound)
        | (cleaned_df[col] > upper_bound)
    )

    outlier_results.append({
        "Column": col,
        "Lower_Bound": round(lower_bound, 2),
        "Upper_Bound": round(upper_bound, 2),
        "Outlier_Count": int(outlier_mask.sum())
    })

outlier_report = pd.DataFrame(outlier_results)
print(outlier_report.to_string(index=False))

# Outliers are flagged, not automatically deleted.
# A high shipping cost can be genuine for a long-distance shipment.

# ------------------------------------------------------------
# STEP 12: Feature engineering
# ------------------------------------------------------------

print("\n15. FEATURE ENGINEERING")

# Calculate delivery duration only when both dates exist
if {"Order_Date", "Actual_Delivery"}.issubset(cleaned_df.columns):
    cleaned_df["Delivery_Days"] = (
        cleaned_df["Actual_Delivery"] - cleaned_df["Order_Date"]
    ).dt.days

# Calculate dispatch duration if dates are available
if {"Order_Date", "Dispatch_Date"}.issubset(cleaned_df.columns):
    cleaned_df["Dispatch_Days"] = (
        cleaned_df["Dispatch_Date"] - cleaned_df["Order_Date"]
    ).dt.days

# ------------------------------------------------------------
# STEP 13: Normalize numerical data
# ------------------------------------------------------------

print("\n16. NUMERICAL NORMALIZATION")

# Min-Max normalization converts values to a 0-to-1 scale.
# Original numeric columns remain unchanged.
normalization_columns = [
    col for col in numeric_columns
    if col in cleaned_df.columns
]

for col in normalization_columns:
    min_value = cleaned_df[col].min()
    max_value = cleaned_df[col].max()

    if pd.notna(min_value) and pd.notna(max_value):
        if max_value != min_value:
            cleaned_df[col + "_Normalized"] = (
                (cleaned_df[col] - min_value)
                / (max_value - min_value)
            )
        else:
            cleaned_df[col + "_Normalized"] = (
                0.0 if pd.notna(min_value) else np.nan
            )

print("Normalized columns created where possible.")

# ------------------------------------------------------------
# STEP 14: Save cleaned dataset and reports
# ------------------------------------------------------------

print("\n17. SAVE OUTPUT FILES")

cleaned_file = os.path.join(
    OUTPUT_DIR, "week2_cleaned_logistics_data.csv"
)
cleaned_df.to_csv(cleaned_file, index=False)

missing_report.to_csv(
    os.path.join(OUTPUT_DIR, "week2_missing_values_report.csv"),
    index=False
)

outlier_report.to_csv(
    os.path.join(OUTPUT_DIR, "week2_outlier_report.csv"),
    index=False
)

validation_report.to_csv(
    os.path.join(OUTPUT_DIR, "week2_validation_report.csv"),
    index=False
)

# ------------------------------------------------------------
# STEP 15: Generate a data quality summary
# ------------------------------------------------------------

quality_summary = pd.DataFrame([
    {
        "Metric": "Original row count",
        "Value": len(df)
    },
    {
        "Metric": "Original column count",
        "Value": len(df.columns)
    },
    {
        "Metric": "Exact duplicate rows detected",
        "Value": duplicate_count
    },
    {
        "Metric": "Rows after duplicate removal",
        "Value": len(cleaned_df)
    },
    {
        "Metric": "Missing cells before cleaning",
        "Value": int(df.isnull().sum().sum())
    },
    {
        "Metric": "Missing cells after preprocessing",
        "Value": int(cleaned_df.isnull().sum().sum())
    }
])

quality_summary.to_csv(
    os.path.join(OUTPUT_DIR, "week2_data_quality_summary.csv"),
    index=False
)

print("\n18. FINAL SUMMARY")
print(quality_summary.to_string(index=False))

print("\nAll Week 2 output files have been saved in:", OUTPUT_DIR)
print("Preprocessing completed successfully!")
