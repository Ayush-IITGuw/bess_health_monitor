"""This stage applies the Isolation Forest anomaly detection algorithm to the dataset after exploratory data analysis (EDA) and
feature engineering have been completed. Rather than operating on raw sensor measurements, the model is trained on derived features
that encode battery behavior, thermal dynamics, and efficiency patterns at a one‑minute resolution."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


iso = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)



