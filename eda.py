import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Load all trace files
trace_files = ['data/Drive_Me_Not/trace'+ str(i) + '.csv' for i in range(1, 9)]
df_list = [pd.read_csv(file) for file in trace_files]
df = pd.concat(df_list, ignore_index=True)

# Convert Time column to readable format if needed
df["Time"] = pd.to_datetime(df["Time"], unit="ms")

# Display basic info
print("Dataset Overview:")
print(df.info())
print("\nFirst 5 Rows:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:\n", df.describe())

# ---------------------- 1. GPS Heatmap ----------------------
map_center = [df["GPS_lat"].mean(), df["GPS_long"].mean()]
m = folium.Map(location=map_center, zoom_start=12)

heat_data = list(zip(df["GPS_lat"], df["GPS_long"]))
HeatMap(heat_data).add_to(m)

m.save("gps_heatmap.html")
print("\n🌍 GPS Heatmap saved as gps_heatmap.html. Open it in a browser to view.")

# ---------------------- 2. Signal Strength (dBm) Analysis ----------------------
plt.figure(figsize=(10, 5))
sns.histplot(df["dBm"], bins=30, kde=True, color="blue")
plt.axvline(df["dBm"].mean(), color="red", linestyle="dashed", label="Mean Signal Strength")
plt.title("Signal Strength (dBm) Distribution")
plt.xlabel("Signal Strength (dBm)")
plt.ylabel("Count")
plt.legend()
plt.show()

# ---------------------- 3. Registered vs. Non-Registered Towers ----------------------
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Registered"], palette="coolwarm")
plt.title("Registered vs. Non-Registered Towers")
plt.xlabel("Registered (True/False)")
plt.ylabel("Count")
plt.show()

# ---------------------- 4. Network Type Distribution ----------------------
plt.figure(figsize=(8, 5))
sns.countplot(y=df["Type"], palette="viridis")
plt.title("Network Type Distribution")
plt.xlabel("Count")
plt.ylabel("Network Type")
plt.show()

# ---------------------- 5. Time-Series Analysis of Signal Strength ----------------------
plt.figure(figsize=(12, 5))
sns.lineplot(x=df["Time"], y=df["dBm"], hue=df["Registered"], palette="coolwarm", alpha=0.7)
plt.title("Time-Series Analysis of Signal Strength")
plt.xlabel("Time")
plt.ylabel("Signal Strength (dBm)")
plt.xticks(rotation=45)
plt.show()

# ---------------------- 6. Signal Strength vs. Cell ID ----------------------
plt.figure(figsize=(12, 5))
sns.boxplot(x=df["CID"], y=df["dBm"], palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Signal Strength per Cell ID")
plt.xlabel("Cell ID")
plt.ylabel("Signal Strength (dBm)")
plt.show()
