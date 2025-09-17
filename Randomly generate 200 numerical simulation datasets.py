# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----- Step 1: Generate Cartesian dataset -----
gm_types = ["Imperial-Vally-06", "Imperial-Vally-060", "Landers", "Menyuan", "Wenchuan"]
pga_levels = np.linspace(0.1, 1.2, 40)  # 40 levels → 5*40 = 200 rows

df = pd.DataFrame(list(product(gm_types, pga_levels)), columns=["GM_Type", "PGA_g"])

# Save TXT file (tab-separated)
out_txt = Path("ground_motion_cartesian_200.txt")
df.to_csv(out_txt, sep="\t", index=False)
print(f"TXT saved to: {out_txt.resolve()}")

# ----- Step 2: Create synthetic target -----
type_weight = {
    "Imperial-Vally-06": 1.00,
    "Imperial-Vally-060": 0.95,
    "Landers":           1.10,
    "Menyuan":           0.90,
    "Wenchuan":          1.25,
}
beta = 1.5
rng = np.random.default_rng(42)
noise = rng.normal(0.0, 0.05, size=len(df))
y = df["GM_Type"].map(type_weight).values * (df["PGA_g"].values ** beta) * 10.0 + noise

# ----- Step 3: Train Random Forest -----
X = df[["GM_Type", "PGA_g"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["GM_Type"]),
        ("num", "passthrough", ["PGA_g"]),
    ]
)
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

model = Pipeline(steps=[("prep", preprocess), ("rf", rf)])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ----- Step 4: Evaluate -----
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE

print(f"Random Forest performance:")
print(f"  R^2  = {r2:.3f}")
print(f"  MAE  = {mae:.3f}")
print(f"  RMSE = {rmse:.3f}")
