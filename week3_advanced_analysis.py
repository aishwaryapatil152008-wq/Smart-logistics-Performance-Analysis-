
# ============================================================
# PROJECT: Smart Logistics Performance Analysis Using Python
# WEEK 3: Advanced Data Analysis and Visualization
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. SET FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = "outputs/week2_cleaned_logistics_data.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD THE CLEANED DATASET
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Check that the Week 2 cleaned CSV exists in outputs."
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("WEEK 3: ADVANCED LOGISTICS DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 3. PREPARE DATA TYPES
# ------------------------------------------------------------

date_columns = [
    "Order_Date",
    "Dispatch_Date",
    "Expected_Delivery",
    "Actual_Delivery"
]

for column in date_columns:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

numeric_columns = [
    "Distance_KM",
    "Shipping_Cost",
    "Weight_KG"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

required_columns = [
    "Shipment_ID",
    "Origin",
    "Destination",
    "Distance_KM",
    "Shipping_Cost",
    "Weight_KG",
    "Dispatch_Date",
    "Expected_Delivery",
    "Actual_Delivery"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# ------------------------------------------------------------
# 4. CALCULATE DELIVERY TIME AND DELAY
# ------------------------------------------------------------

df["Delivery_Days"] = (
    df["Actual_Delivery"] - df["Dispatch_Date"]
).dt.days

df["Delay_Days"] = (
    df["Actual_Delivery"] - df["Expected_Delivery"]
).dt.days

# ------------------------------------------------------------
# 5. CLASSIFY DELIVERY PERFORMANCE
# ------------------------------------------------------------

df["Performance_Status"] = "Pending"

delivered = df["Actual_Delivery"].notna()

df.loc[
    delivered & (df["Delay_Days"] <= 0),
    "Performance_Status"
] = "On Time"

df.loc[
    delivered & (df["Delay_Days"] > 0),
    "Performance_Status"
] = "Late"

# ------------------------------------------------------------
# 6. CALCULATE KEY PERFORMANCE INDICATORS (KPIs)
# ------------------------------------------------------------

total_shipments = len(df)
delivered_shipments = int(delivered.sum())
pending_shipments = total_shipments - delivered_shipments

average_cost = df["Shipping_Cost"].mean()
average_distance = df["Distance_KM"].mean()
average_weight = df["Weight_KG"].mean()
average_delivery_days = df["Delivery_Days"].mean()

on_time_count = int(
    (df["Performance_Status"] == "On Time").sum()
)

late_count = int(
    (df["Performance_Status"] == "Late").sum()
)

on_time_rate = (
    on_time_count / delivered_shipments * 100
    if delivered_shipments > 0
    else 0
)

print("\n" + "-" * 45)
print("KEY PERFORMANCE INDICATORS")
print("-" * 45)

print("Total shipments:", total_shipments)
print("Delivered shipments:", delivered_shipments)
print("Pending shipments:", pending_shipments)
print(f"Average shipping cost: {average_cost:.2f}")
print(f"Average distance: {average_distance:.2f} km")
print(f"Average shipment weight: {average_weight:.2f} kg")
print(f"Average delivery time: {average_delivery_days:.2f} days")
print(f"On-time delivery rate: {on_time_rate:.2f}%")
print("Late shipments:", late_count)

# ------------------------------------------------------------
# 7. ROUTE-WISE ANALYSIS
# ------------------------------------------------------------

route_summary = (
    df.groupby(["Origin", "Destination"], dropna=False)
    .agg(
        Shipment_Count=("Shipment_ID", "count"),
        Average_Cost=("Shipping_Cost", "mean"),
        Average_Distance=("Distance_KM", "mean"),
        Average_Weight=("Weight_KG", "mean"),
        Average_Delivery_Days=("Delivery_Days", "mean"),
        Late_Shipments=(
            "Performance_Status",
            lambda x: (x == "Late").sum()
        )
    )
    .reset_index()
)

route_summary.to_csv(
    os.path.join(OUTPUT_DIR, "week3_route_analysis.csv"),
    index=False
)

print("\nROUTE-WISE ANALYSIS")
print(route_summary.to_string(index=False))

# ------------------------------------------------------------
# 8. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

analysis_columns = [
    "Distance_KM",
    "Shipping_Cost",
    "Weight_KG",
    "Delivery_Days",
    "Delay_Days"
]

statistics = df[analysis_columns].describe().T

statistics.to_csv(
    os.path.join(OUTPUT_DIR, "week3_descriptive_statistics.csv")
)

print("\nDESCRIPTIVE STATISTICS")
print(statistics.round(2))

# ------------------------------------------------------------
# 9. CORRELATION ANALYSIS
# ------------------------------------------------------------

correlation = df[analysis_columns].corr()

correlation.to_csv(
    os.path.join(OUTPUT_DIR, "week3_correlation_matrix.csv")
)

print("\nCORRELATION MATRIX")
print(correlation.round(2))

# ------------------------------------------------------------
# 10. COST AND DELIVERY PERFORMANCE SUMMARY
# ------------------------------------------------------------

cost_summary = (
    df.groupby("Performance_Status", dropna=False)
    .agg(
        Shipment_Count=("Shipment_ID", "count"),
        Average_Shipping_Cost=("Shipping_Cost", "mean"),
        Average_Distance=("Distance_KM", "mean"),
        Average_Delivery_Days=("Delivery_Days", "mean")
    )
    .reset_index()
)

cost_summary.to_csv(
    os.path.join(OUTPUT_DIR, "week3_cost_performance_summary.csv"),
    index=False
)

# ------------------------------------------------------------
# 11. GRAPH 1: DELIVERY PERFORMANCE
# ------------------------------------------------------------

status_counts = df["Performance_Status"].value_counts()

plt.figure(figsize=(8, 5))
status_counts.plot(kind="bar")

plt.title("Logistics Delivery Performance")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Shipments")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "week3_delivery_performance.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 12. GRAPH 2: SHIPPING COST VS DISTANCE
# ------------------------------------------------------------

plot_data = df.dropna(
    subset=["Distance_KM", "Shipping_Cost"]
)

plt.figure(figsize=(8, 5))
plt.scatter(
    plot_data["Distance_KM"],
    plot_data["Shipping_Cost"]
)

plt.title("Shipping Cost vs Distance")
plt.xlabel("Distance (KM)")
plt.ylabel("Shipping Cost")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "week3_cost_vs_distance.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 13. GRAPH 3: AVERAGE COST BY ROUTE
# ------------------------------------------------------------

route_labels = (
    route_summary["Origin"].astype(str)
    + " to "
    + route_summary["Destination"].astype(str)
)

plt.figure(figsize=(10, 6))
plt.bar(route_labels, route_summary["Average_Cost"])

plt.title("Average Shipping Cost by Route")
plt.xlabel("Route")
plt.ylabel("Average Shipping Cost")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "week3_route_shipping_cost.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 14. GRAPH 4: CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation Coefficient")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

plt.title("Correlation Between Logistics Variables")

for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        value = correlation.iloc[i, j]

        if pd.notna(value):
            plt.text(
                j, i, f"{value:.2f}",
                ha="center", va="center"
            )

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "week3_correlation_heatmap.png"),
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 15. SAVE ANALYZED DATASET
# ------------------------------------------------------------

df.to_csv(
    os.path.join(OUTPUT_DIR, "week3_analyzed_shipments.csv"),
    index=False
)

# ------------------------------------------------------------
# 16. SAVE KPI REPORT
# ------------------------------------------------------------

kpi_summary = pd.DataFrame({
    "Metric": [
        "Total Shipments",
        "Delivered Shipments",
        "Pending Shipments",
        "Average Shipping Cost",
        "Average Distance KM",
        "Average Weight KG",
        "Average Delivery Days",
        "On-Time Delivery Rate Percent",
        "Late Shipments"
    ],
    "Value": [
        total_shipments,
        delivered_shipments,
        pending_shipments,
        average_cost,
        average_distance,
        average_weight,
        average_delivery_days,
        on_time_rate,
        late_count
    ]
})

kpi_summary.to_csv(
    os.path.join(OUTPUT_DIR, "week3_kpi_summary.csv"),
    index=False
)

# ------------------------------------------------------------
# 17. FINISH
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("WEEK 3 ANALYSIS COMPLETED SUCCESSFULLY!")
print("All generated reports and graphs are in the outputs folder.")
print("=" * 60)
