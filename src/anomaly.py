"""This stage applies the Isolation Forest anomaly detection algorithm to the dataset after exploratory data analysis (EDA) and
feature engineering have been completed. Rather than operating on raw sensor measurements, the model is trained on derived features
that encode battery behavior, thermal dynamics, and efficiency patterns at a one‑minute resolution."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# feature csv generated from eda
csv_path = "outputs/eda/45min/window_features_45min.csv"

df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
print(df.shape)
df.head()

# Relevant features for isolation forest
features = [
    "deltaT_45min_end_minus_start",
    "temp_rise_per_kWh_45min",
    "Edis_45min_kWh",
]

X = df[features].copy()
X = X.dropna()

# Scale the features before applying isolation forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Instantiation of IsolationForest Model
iso = IsolationForest(
    n_estimators=300,
    contamination=0.01,
    random_state=42
)
labels = iso.fit_predict(X_scaled)
X["anomaly_flag"] = labels
X["is_anomaly"] = X["anomaly_flag"] == -1
df_anomaly = df.loc[X.index].copy()
df_anomaly["is_anomaly"] = X["is_anomaly"]

out_csv = "outputs/eda/45min/window_features_45min_with_anomalies.csv"
df_anomaly.to_csv(out_csv)
print("[WRITE]", out_csv)



