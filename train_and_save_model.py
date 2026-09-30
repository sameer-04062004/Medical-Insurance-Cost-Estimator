import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():
    csv_path = "insurance.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    initial_shape = df.shape
    df = df.drop_duplicates(keep="first")
    print(f"Loaded {df.shape[0]} records after removing duplicates (was {initial_shape[0]}).")

    # One-hot encoding exactly as done in the notebook
    df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)
    
    X = df_encoded.drop(columns=["charges"])
    y_dollars = df_encoded["charges"]
    y_log = np.log(y_dollars)

    feature_cols = list(X.columns)
    print("Features:", feature_cols)

    # Train/test split matching the notebook
    X_train, X_test, y_train_log, y_test_log, y_train_usd, y_test_usd = train_test_split(
        X, y_log, y_dollars, test_size=0.2, random_state=42
    )

    # 1. Gradient Boosting Model (Best performing model from notebook)
    boost_model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    boost_model.fit(X_train, y_train_log)

    pred_log_boost = boost_model.predict(X_test)
    pred_usd_boost = np.exp(pred_log_boost)

    r2_boost = r2_score(y_test_usd, pred_usd_boost)
    mae_boost = mean_absolute_error(y_test_usd, pred_usd_boost)
    rmse_boost = np.sqrt(mean_squared_error(y_test_usd, pred_usd_boost))

    print(f"--- Gradient Boosting Model Results ---")
    print(f"R2 Score (Dollars): {r2_boost:.4f} (~{r2_boost*100:.1f}%)")
    print(f"MAE:  ${mae_boost:,.2f}")
    print(f"RMSE: ${rmse_boost:,.2f}")

    # 2. Linear Baseline Model (For optional comparison)
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train_log)
    pred_log_base = baseline_model.predict(X_test)
    pred_usd_base = np.exp(pred_log_base)
    r2_base = r2_score(y_test_usd, pred_usd_base)
    mae_base = mean_absolute_error(y_test_usd, pred_usd_base)

    # Summary statistics for reference in UI
    stats = {
        "mean_charges": float(df["charges"].mean()),
        "median_charges": float(df["charges"].median()),
        "min_charges": float(df["charges"].min()),
        "max_charges": float(df["charges"].max()),
        "smoker_mean": float(df[df["smoker"] == "yes"]["charges"].mean()),
        "non_smoker_mean": float(df[df["smoker"] == "no"]["charges"].mean()),
    }

    # Model package dictionary
    model_bundle = {
        "model": boost_model,
        "baseline_model": baseline_model,
        "feature_names": feature_cols,
        "metrics": {
            "r2": r2_boost,
            "mae": mae_boost,
            "rmse": rmse_boost,
            "baseline_r2": r2_base,
            "baseline_mae": mae_base
        },
        "stats": stats,
        "target_transform": "log_natural",
    }

    # Save to file
    output_path = "insurance_cost_model.joblib"
    joblib.dump(model_bundle, output_path)
    print(f"\n[SUCCESS] Model bundle successfully saved to '{output_path}'")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
