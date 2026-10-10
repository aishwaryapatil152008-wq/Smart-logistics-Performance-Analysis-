
# ============================================================
# PROJECT: Smart Logistics Performance Analysis Using Python
# WEEK 3: Advanced Data Analysis and Visualization
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define input and output paths
INPUT_FILE = "outputs/week2_cleaned_logistics_data.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load the cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 55)
print("WEEK 3: ADVANCED LOGISTICS DATA ANALYSIS")
print("=" * 55)

print("\nDataset shape:", df.shape)
print("\nDataset columns:")
print(df.columns.tolist())

# 3. Convert date columns into datetime format
date_columns = [
    "Order_Date",
    "Dispatch_Date",
    "Expected_Delivery",
    "Actual_Delivery"
]

for column in date_columns:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

# 4. Convert numeric columns
numeric_columns = ["Distance_KM", "Shipping_Cost", "Weight_KG"]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# 5. Calculate delivery time and delay
# Missing actual delivery dates remain missing.
df["Delivery_Days"] = (
    df["Actual_Delivery"] - df["Dispatch_Date"]
).dt.days

df["Delay_Days"] = (
    df["Actual_Delivery"] - df["Expected_Delivery"]
).dt.days

# 6. Identify on-time and late deliveries
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

# 7. Calculate key performance indicators (KPIs)
total_shipments = len(df)
delivered_shipments = int(delivered.sum())
pending_shipments = total_shipments - delivered_shipments

average_shipping_cost = df["Shipping_Cost"].mean()
average_distance = df["Distance_KM"].mean()
average_weight = df["Weight_KG"].mean()
average_delivery_days = df["Delivery_Days"].mean()

on_time_count = int((df["Performance_Status"] == "On Time").sum())
late_count = int((df["Performance_Status"] == "Late").sum())

on_time_rate = (
    on_time_count / delivered_shipments * 100
    if delivered_shipments else 0
)

print("\n--- KEY PERFORMANCE INDICATORS ---")
print("Total shipments:", total_shipments)
print("Delivered shipments:", delivered_shipments)
print("Pending shipments:", pending_shipments)
print(f"Average shipping cost: {average_shipping_cost:.2f}")
print(f"Average distance: {average_distance:.2f} km")
print(f"Average weight: {average_weight:.2f} kg")
print(f"Average delivery time: {average_delivery_days:.2f} days")
print(f"On-time delivery rate: {on_time_rate:.2f}%")
print("Late deliveries:", late_count)

# 8. Analyze delivery performance by route
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

# 9. Analyze shipping cost by delivery status
cost_summary = (
    df.groupby("Performance_Status", dropna=False)
    .agg(
        Shipment_Count=("Shipment_ID", "count"),
        Average_Shipping_Cost=("Shipping_Cost", "mean"),
        Average_Distance=("Distance_KM", "mean")
    )
    .reset_index()
)

cost_summary.to_csv(
    os.path.join(OUTPUT_DIR, "week3_cost_performance_summary.csv"),
    index=False
)

# 10. Calculate descriptive statistics
statistics = df[
    ["Distance_KM", "Shipping_Cost", "Weight_KG",
     "Delivery_Days", "Delay_Days"]
].describe().T

statistics.to_csv(
    os.path.join(OUTPUT_DIR, "week3_descriptive_statistics.csv")
)

# 11. Calculate correlations between numeric variables
correlation_columns = [
    "Distance_KM",
    "Shipping_Cost",
    "Weight_KG",
    "Delivery_Days",
    "Delay_Days"
]

correlation = df[correlation_columns].corr()
correlation.to_csv(
    os.path.join(OUTPUT_DIR, "week3_correlation_matrix.csv")
)

print("\n--- CORRELATION MATRIX ---")
print(correlation.round(2))

# 12. Visualization 1: Shipment delivery status
status_counts = df["Performance_Status"].value_counts()

plt.figure(figsize=(8, 5))
status_counts.plot(kind="bar")
plt.title("Logistics Shipment Performance Status")
plt.xlabel("Performance Status")
plt.ylabel("Number of Shipments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "week3_delivery_performance.png"),
    dpi=300
)
plt.close()

# 13. Visualization 2: Shipping cost versus distance
plot_data = df.dropna(subset=["Distance_KM", "Shipping_Cost"])

plt.figure(figsize=(8, 5))
plt.scatter(
    plot_data["Distance_KM"],
    plot_data["Shipping_Cost"]
)
plt.title("Shipping Cost vs Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Shipping Cost")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "week3_cost_vs_distance.png"),
    dpi=300
)
plt.close()

# 14. Visualization 3: Average shipping cost by route
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

# 15. Visualization 4: Correlation heatmap using Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation Coefficient")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(correlation.index)), correlation.index)
plt.title("Correlation Between Logistics Variables")

for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        value = correlation.iloc[i, j]
        if pd.notna(value):
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "week3_correlation_heatmap.png"),
    dpi=300
)
plt.close()

# 16. Save shipment-level analysis
df.to_csv(
    os.path.join(OUTPUT_DIR, "week3_analyzed_shipments.csv"),
    index=False
)

# 17. Save KPI summary
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
        average_shipping_cost,
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

print("\nAnalysis completed successfully!")
print("Graphs and reports saved in the outputs folder.")
print("\nPlease review the generated reports before interpreting results.")
