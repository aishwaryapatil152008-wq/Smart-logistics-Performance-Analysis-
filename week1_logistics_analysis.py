
# ============================================================
# PROJECT: SMART LOGISTICS PERFORMANCE ANALYSIS USING PYTHON
# Description: Clean logistics data, calculate KPIs, and
# generate four charts.
# ============================================================

import os
import sys
import pandas as pd
import matplotlib

# Use a non-interactive backend so charts save correctly
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows terminal encoding issues
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# STEP 1: CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join("data", "logistics_data.csv")

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

print("\n========== SMART LOGISTICS ANALYSIS ==========\n")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Make sure logistics_data.csv is inside the data folder."
    )

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("\n========== FIRST FIVE SHIPMENTS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# ============================================================
# STEP 3: CHECK AND CLEAN DATA
# ============================================================

print("\n========== DATA QUALITY CHECK ==========")
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
clean_df = df.drop_duplicates().copy()

# Convert date columns to datetime where present
date_columns = [
    "Order_Date",
    "Dispatch_Date",
    "Expected_Delivery",
    "Actual_Delivery"
]

for column in date_columns:
    if column in clean_df.columns:
        clean_df[column] = pd.to_datetime(
            clean_df[column], errors="coerce"
        )

# Convert numeric columns to numeric values where present
numeric_columns = [
    "Shipping_Cost",
    "Distance_KM",
    "Delivery_Days",
    "Weight_KG"
]

for column in numeric_columns:
    if column in clean_df.columns:
        clean_df[column] = pd.to_numeric(
            clean_df[column], errors="coerce"
        )

print("\nData cleaning completed!")
print("Rows after removing duplicates:", len(clean_df))

# Save cleaned data
clean_df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_logistics_data.csv"),
    index=False
)

# Calculate delivery time using the actual dataset columns

clean_df["Order_Date"] = pd.to_datetime(
    clean_df["Order_Date"], errors="coerce"
)

clean_df["Actual_Delivery"] = pd.to_datetime(
    clean_df["Actual_Delivery"], errors="coerce"
)

clean_df["Delivery_Days"] = (
    clean_df["Actual_Delivery"] - clean_df["Order_Date"]
).dt.days

print("\nDelivery time calculated successfully!")
print(clean_df[["Shipment_ID", "Delivery_Days"]].head())

# ============================================================
# STEP 4: CALCULATE KEY PERFORMANCE INDICATORS (KPIs)
# ============================================================

required_columns = [
    "Shipment_ID",
    "Destination",
    "Shipping_Cost",
    "Delivery_Status",
    "Distance_KM",
    
]

missing_columns = [
    column for column in required_columns
    if column not in clean_df.columns
]

if missing_columns:
    raise ValueError(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )

# Standardize status text for analysis
status = clean_df["Delivery_Status"].astype(str).str.strip().str.lower()

in_transit_mask = status.str.contains(
    "transit|in progress|pending", na=False
)
on_time_mask = status.str.contains(
    "on.?time|on time", na=False
)
delayed_mask = status.str.contains(
    "delay|late", na=False
)

total_shipments = len(clean_df)
in_transit_shipments = int(in_transit_mask.sum())
on_time_shipments = int(on_time_mask.sum())
delayed_shipments = int(delayed_mask.sum())

# Count shipments that are no longer in transit
delivered_shipments = total_shipments - in_transit_shipments

on_time_rate = (
    on_time_shipments / delivered_shipments * 100
    if delivered_shipments else 0
)

delay_rate = (
    delayed_shipments / delivered_shipments * 100
    if delivered_shipments else 0
)

average_delivery_time = clean_df["Delivery_Days"].mean()
average_shipping_cost = clean_df["Shipping_Cost"].mean()

total_distance = clean_df["Distance_KM"].sum()
total_shipping_cost = clean_df["Shipping_Cost"].sum()

average_cost_per_km = (
    total_shipping_cost / total_distance
    if total_distance else 0
)

kpis = {
    "Total Shipments": total_shipments,
    "Delivered Shipments": delivered_shipments,
    "In-Transit Shipments": in_transit_shipments,
    "On-Time Shipments": on_time_shipments,
    "Delayed Shipments": delayed_shipments,
    "On-Time Delivery Rate (%)": round(on_time_rate, 2),
    "Delay Rate (%)": round(delay_rate, 2),
    "Average Delivery Time (Days)": round(average_delivery_time, 2),
    "Average Shipping Cost (Rs.)": round(average_shipping_cost, 2),
    "Average Shipping Cost per KM (Rs.)": round(average_cost_per_km, 2)
}

print("\n========== LOGISTICS KPIs ==========")

for name, value in kpis.items():
    print(f"{name}: {value}")

# Save KPIs as a CSV report
pd.DataFrame(
    list(kpis.items()),
    columns=["KPI", "Value"]
).to_csv(
    os.path.join(OUTPUT_DIR, "logistics_kpi_report.csv"),
    index=False
)

# ============================================================
# STEP 5: SET GRAPH STYLE
# ============================================================

sns.set_theme(style="whitegrid")

# CHART 1: AVERAGE SHIPPING COST BY DESTINATION
# ============================================================

print("\nGenerating Chart 1...")

destination_cost = (
    clean_df.groupby("Destination")["Shipping_Cost"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
destination_cost.plot(kind="bar")
plt.title("Average Shipping Cost by Destination")
plt.xlabel("Destination")
plt.ylabel("Average Shipping Cost (Rs.)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "shipping_cost_by_destination.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Chart 1 saved successfully!")

# CHART 2: DELIVERY STATUS DISTRIBUTION
# ============================================================

print("Generating Chart 2...")

status_counts = clean_df["Delivery_Status"].fillna(
    "Unknown"
).value_counts()

plt.figure(figsize=(8, 6))
plt.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Delivery Status Distribution")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "delivery_status_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Chart 2 saved successfully!")

# CHART 3: DELIVERY TIME BY SHIPMENT
# ============================================================

print("Generating Chart 3...")

delivered_df = clean_df.dropna(
    subset=["Delivery_Days"]
).copy()

if not delivered_df.empty:
    if "Order_Date" in delivered_df.columns:
        delivered_df = delivered_df.sort_values("Order_Date")

    plt.figure(figsize=(12, 6))
    plt.plot(
        delivered_df["Shipment_ID"].astype(str),
        delivered_df["Delivery_Days"],
        marker="o",
        linewidth=2
    )
    plt.title("Delivery Time by Shipment")
    plt.xlabel("Shipment ID")
    plt.ylabel("Delivery Time (Days)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "delivery_time_by_shipment.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Chart 3 saved successfully!")
else:
    print("Chart 3 skipped: no delivery-time data is available.")

# CHART 4: DISTANCE VS SHIPPING COST
# ============================================================

print("Generating Chart 4...")

scatter_df = clean_df.dropna(
    subset=["Distance_KM", "Shipping_Cost"]
).copy()

if not scatter_df.empty:
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=scatter_df,
        x="Distance_KM",
        y="Shipping_Cost",
        hue="Delivery_Status",
        s=100
    )

    plt.title("Distance vs Shipping Cost")
    plt.xlabel("Distance (KM)")
    plt.ylabel("Shipping Cost (Rs.)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "distance_vs_shipping_cost.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Chart 4 saved successfully!")
else:
    print("Chart 4 skipped: no distance and cost data is available.")

# ============================================================
# STEP 6: FINAL SUMMARY
# ============================================================

print("\n========== PROJECT COMPLETED ==========")
print("Cleaned dataset saved in:", OUTPUT_DIR)
print("KPI report saved in:", OUTPUT_DIR)
print("Charts folder:", os.path.abspath(OUTPUT_DIR))

print("\nFiles generated:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nOpen the outputs folder to view your results.")
