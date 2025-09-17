# -*- coding: utf-8 -*-
"""
AdaBoost.RT for resilience evaluation of high-speed railway tunnels
"""

import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone

# ----------------------------
# Config
# ----------------------------
DATA_TXT = Path(r"C:\Users\Administrator\Desktop\my_data.txt")
SEED = 42
N_SPLITS_OUTER = 10
TRAIN_RATIO = 0.7
CV_FOLDS = 5
PATIENCE = 10
L2_LAMBDA = 0.01
PHI = 0.10
T_MAX = 100

DAMAGE_STATE_THRESHOLDS = {
    "slight": 1.0,
    "moderate": 1.5,
    "extensive": 2.5,
    "complete": 3.5
}


def ensure_data_file(p: Path) -> Path:
    if p.exists():
        return p
    rng = np.random.default_rng(2025)
    n = 200
    M = rng.uniform(150, 600, size=n)           # kN·m
    D = rng.uniform(0.5, 5.0, size=n)           # mm
    PGA = rng.uniform(0.1, 1.2, size=n)         # g
    # 构造一个示例 DI（与 M/D、PGA 有关）
    DI = 0.25 * (M / D) * (1.0 + 0.2 * PGA) + rng.normal(0, 5.0, size=n)

    df = pd.DataFrame({
        "M": np.round(M, 3),
        "D": np.round(D, 3),
        "PGA": np.round(PGA, 3),
        "DI": np.round(DI, 3),
    })
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False)
    print(f"[Info] Data file not found. A demo dataset has been generated at:\n{p}")
    return p

# ----------------------------
# ELM Regressor (ridge)
# ----------------------------
class ELMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=200, alpha=1e-2, random_state=None):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state
        self.W_ = None
        self.b_ = None
        self.beta_ = None
        self.scaler_ = StandardScaler()

    def _hidden(self, X):
        Z = X @ self.W_.T + self.b_
        return np.tanh(Z)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        Xs = self.scaler_.fit_transform(X)
        n_features = Xs.shape[1]
        self.W = rng.normal(0, 1, size=(self.n_hidden, n_features))
        self.b = rng.normal(0, 1, size=(self.n_hidden,))
        self.W_ = self.W
        self.b_ = self.b
        H = self._hidden(Xs)
        ridge = Ridge(alpha=self.alpha, fit_intercept=False, solver='auto')
        ridge.fit(H, y)
        self.beta_ = ridge.coef_
        if self.beta_.ndim == 1:
            self.beta_ = self.beta_.reshape(-1,)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(X)
        H = self._hidden(Xs)
        return H @ self.beta_

# ----------------------------
# Base learners
# ----------------------------
def make_base_learners(random_state:int) -> Dict[str, BaseEstimator]:
    base = {
        "SVR": SVR(C=1.0 / L2_LAMBDA, epsilon=0.01, kernel='rbf', gamma='scale'),
        "BP":  MLPRegressor(hidden_layer_sizes=(128, 64),
                            activation='relu',
                            alpha=L2_LAMBDA,
                            learning_rate_init=1e-3,
                            max_iter=5000,
                            early_stopping=True,
                            n_iter_no_change=20,
                            random_state=random_state),
        "ELM": ELMRegressor(n_hidden=300, alpha=L2_LAMBDA, random_state=random_state)
    }
    wrapped = {}
    for name, est in base.items():
        wrapped[name] = Pipeline([
            ("scaler", StandardScaler()),
            ("est", est)
        ])
    return wrapped

# ----------------------------
# AdaBoost.RT
# ----------------------------
@dataclass
class AdaBoostRTConfig:
    base_estimator: BaseEstimator
    n_estimators: int = 100
    phi: float = 0.1
    learning_rate: float = 1.0
    early_stopping: bool = True
    patience: int = 10
    random_state: Optional[int] = None

class AdaBoostRT(BaseEstimator, RegressorMixin):
    def __init__(self, cfg: AdaBoostRTConfig):
        self.cfg = cfg
        self.estimators_ = []
        self.alphas_ = []
        self.best_iter_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        n = X.shape[0]
        D = np.full(n, 1.0 / n)
        best_val = np.inf
        bad_rounds = 0

        for t in range(self.cfg.n_estimators):
            est = clone(self.cfg.base_estimator)
            try:
                est.fit(X, y, **{"est__sample_weight": D})
            except Exception:
                try:
                    est.fit(X, y, sample_weight=D)
                except Exception:
                    est.fit(X, y)

            y_hat = est.predict(X)
            denom = np.maximum(np.abs(y), 1e-8)
            E = np.abs(y - y_hat) / denom
            bad = (E > self.cfg.phi).astype(float)
            eps_t = float(np.sum(D * bad))
            eps_t = min(max(eps_t, 1e-12), 1 - 1e-12)
            alpha_t = np.log((1 - eps_t) / eps_t) * self.cfg.learning_rate

            D = D * np.exp(alpha_t * bad)
            D = D / D.sum()

            self.estimators_.append(est)
            self.alphas_.append(alpha_t)

            if X_val is not None and y_val is not None and self.cfg.early_stopping:
                y_val_pred = self.predict(X_val)
                val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
                if val_rmse + 1e-12 < best_val:
                    best_val = val_rmse
                    bad_rounds = 0
                    self.best_iter_ = t
                else:
                    bad_rounds += 1
                    if bad_rounds >= self.cfg.patience:
                        break

        if self.best_iter_ is None:
            self.best_iter_ = len(self.estimators_) - 1
        return self

    def predict(self, X):
        if not self.estimators_:
            raise RuntimeError("Model not fitted.")
        Tstar = self.best_iter_ + 1 if self.best_iter_ is not None else len(self.estimators_)
        alphas = np.array(self.alphas_[:Tstar])
        w = alphas / (np.sum(np.abs(alphas)) + 1e-12)
        preds = np.column_stack([est.predict(X) for est in self.estimators_[:Tstar]])
        return preds @ w

# ----------------------------
# CV: select base learner + hyper-params
# ----------------------------
def cross_validate_adart(X, y, random_state:int) -> AdaBoostRT:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    base_learners = make_base_learners(random_state)
    grids = {
        "SVR": {"phi":[0.08,0.10,0.12], "n_estimators":[60,100]},
        "BP":  {"phi":[0.08,0.10,0.12], "n_estimators":[60,100]},
        "ELM": {"phi":[0.08,0.10,0.12], "n_estimators":[60,100]}
    }

    best_score = np.inf
    best_cfg = None

    for name, base in base_learners.items():
        for phi in grids[name]["phi"]:
            for Tn in grids[name]["n_estimators"]:
                rmse_list = []
                for tr_idx, val_idx in kf.split(X):
                    Xtr, Xval = X[tr_idx], X[val_idx]
                    ytr, yval = y[tr_idx], y[val_idx]
                    cfg = AdaBoostRTConfig(
                        base_estimator=base,
                        n_estimators=Tn,
                        phi=phi,
                        early_stopping=True,
                        patience=PATIENCE,
                        random_state=random_state
                    )
                    model = AdaBoostRT(cfg).fit(Xtr, ytr, X_val=Xval, y_val=yval)
                    ypred = model.predict(Xval)
                    rmse = mean_squared_error(yval, ypred, squared=False)
                    rmse_list.append(rmse)
                avg_rmse = float(np.mean(rmse_list))
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_cfg = (name, base, Tn, phi)

    name, base, Tn, phi = best_cfg
    final_cfg = AdaBoostRTConfig(
        base_estimator=base,
        n_estimators=Tn,
        phi=phi,
        early_stopping=True,
        patience=PATIENCE,
        random_state=random_state
    )
    return AdaBoostRT(final_cfg)

# ----------------------------
# Fragility curves vs PGA
# ----------------------------
def fit_fragility_curves(PGA: np.ndarray, DI_pred: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    curves = {}
    pga_grid = np.linspace(np.min(PGA), np.max(PGA), 200).reshape(-1,1)
    PGA_col = PGA.reshape(-1,1)
    for state, thr in DAMAGE_STATE_THRESHOLDS.items():
        y_bin = (DI_pred > thr).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(PGA_col, y_bin)
        prob = clf.predict_proba(pga_grid)[:,1]
        curves[state] = (pga_grid.ravel(), prob)
    return curves

# ----------------------------
# Functionality curve Q(t)
# ----------------------------
def functionality_Q_t(T_total=120, t_decision=10, t_repair_start=10, t_repair_end=90,
                      Q_drop=0.4, Q_recover_final=1.0, n_points=200):
    t = np.linspace(0, T_total, n_points)
    Q = np.ones_like(t)
    Q0 = 1.0 - Q_drop
    Q[t >= 0] = Q0
    mask = (t >= t_repair_start) & (t <= t_repair_end)
    tau = (t[mask] - t_repair_start) / max(t_repair_end - t_repair_start, 1e-6)
    Q[mask] = Q0 + (Q_recover_final - Q0) * (1/(1 + np.exp(-8*(tau-0.5))))
    Q[t > t_repair_end] = Q_recover_final
    return t, Q

# ----------------------------
# Main
# ----------------------------
def main():
    data_path = ensure_data_file(DATA_TXT)
    df = pd.read_csv(data_path, sep="\t")

    required = {"M","D","PGA","DI"}
    if not required.issubset(df.columns):
        raise ValueError(f"Data file must contain columns: {required}. Found: {df.columns.tolist()}")

    X = df[["M","D","PGA"]].to_numpy(dtype=float)
    y = df["DI"].to_numpy(dtype=float)

    rng = np.random.default_rng(SEED)
    metrics_val = []
    metrics_test = []
    curves_all_runs = []

    for run in range(N_SPLITS_OUTER):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=TRAIN_RATIO, random_state=int(rng.integers(0, 10_000)), shuffle=True
        )

        model = cross_validate_adart(X_tr, y_tr, random_state=SEED + run)
        X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED + run)
        model.fit(X_tr2, y_tr2, X_val=X_val, y_val=y_val)

        y_val_pred = model.predict(X_val)
        rmse_v = mean_squared_error(y_val, y_val_pred, squared=False)
        mae_v = mean_absolute_error(y_val, y_val_pred)
        r2_v = r2_score(y_val, y_val_pred)
        metrics_val.append((rmse_v, mae_v, r2_v))

        y_te_pred = model.predict(X_te)
        rmse_t = mean_squared_error(y_te, y_te_pred, squared=False)
        mae_t = mean_absolute_error(y_te, y_te_pred)
        r2_t = r2_score(y_te, y_te_pred)
        metrics_test.append((rmse_t, mae_t, r2_t))

        PGA_te = X_te[:, 2]
        curves = fit_fragility_curves(PGA_te, y_te_pred)
        curves_all_runs.append(curves)

    def fmt(mean, std): return f"{mean:.3f} ± {std:.3f}"
    val_arr = np.array(metrics_val)
    te_arr  = np.array(metrics_test)

    print("\n=== Validation (over 10 runs, 5-fold CV inside) ===")
    print(f"RMSE: {fmt(val_arr[:,0].mean(), val_arr[:,0].std())}")
    print(f"MAE : {fmt(val_arr[:,1].mean(), val_arr[:,1].std())}")
    print(f"R²  : {fmt(val_arr[:,2].mean(),  val_arr[:,2].std())}")

    print("\n=== Test (independent, over 10 runs) ===")
    print(f"RMSE: {fmt(te_arr[:,0].mean(), te_arr[:,0].std())}")
    print(f"MAE : {fmt(te_arr[:,1].mean(), te_arr[:,1].std())}")
    print(f"R²  : {fmt(te_arr[:,2].mean(),  te_arr[:,2].std())}")

    t, Q = functionality_Q_t(T_total=120, t_decision=10, t_repair_start=10, t_repair_end=90,
                             Q_drop=0.45, Q_recover_final=1.0)
    pd.DataFrame({"t_day": t, "Q": Q}).to_csv("functionality_Q_t.csv", index=False)
    print("\nSaved functionality curve to functionality_Q_t.csv")

    for state in DAMAGE_STATE_THRESHOLDS.keys():
        grid = np.linspace(df["PGA"].min(), df["PGA"].max(), 200)
        probs = []
        for curves in curves_all_runs:
            if state in curves:
                xg, pg = curves[state]
                probs.append(np.interp(grid, xg, pg))
        if probs:
            avg = np.mean(np.vstack(probs), axis=0)
            pd.DataFrame({"PGA": grid, f"Prob_exceed_{state}": avg}).to_csv(
                f"fragility_{state}.csv", index=False
            )
            print(f"Saved fragility curve for '{state}' to fragility_{state}.csv")

if __name__ == "__main__":
    main()
