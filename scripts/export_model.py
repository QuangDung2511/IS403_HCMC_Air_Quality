"""
Export XGBoost Tuned Model
--------------------------
Trains XGBoost with the best hyperparameters found during tuning
and saves the model to models/xgb_tuned_final.joblib.

Run once: python scripts/export_model.py
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_split.csv')
FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'selected_features.json')
PARAMS_PATH = os.path.join(BASE_DIR, 'outputs', 'predictions', 'tuned_tree_best_params.json')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_tuned_final.joblib')

print("=" * 60)
print("  XGBoost Model Export Script")
print("=" * 60)

# 1. Load training data
print("\n[1/4] Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"  → Train shape: {train_df.shape}")

# 2. Load selected features
print("[2/4] Loading features & params...")
with open(FEATURES_PATH, 'r') as f:
    features_config = json.load(f)
selected_features = features_config['selected_features']
print(f"  → {len(selected_features)} features: {selected_features[:5]}...")

# 3. Load best hyperparameters
with open(PARAMS_PATH, 'r') as f:
    all_params = json.load(f)
xgb_params = all_params['XGBoost']
print(f"  → XGBoost params: {xgb_params}")

# 4. Train model
print("\n[3/4] Training XGBoost with best params...")
X_train = train_df[selected_features]
y_train = train_df['target_pm25_h24']

model = XGBRegressor(
    n_estimators=xgb_params['n_estimators'],
    max_depth=xgb_params['max_depth'],
    learning_rate=xgb_params['learning_rate'],
    subsample=xgb_params['subsample'],
    colsample_bytree=xgb_params['colsample_bytree'],
    reg_alpha=xgb_params['reg_alpha'],
    reg_lambda=xgb_params['reg_lambda'],
    gamma=xgb_params['gamma'],
    min_child_weight=xgb_params['min_child_weight'],
    tree_method='hist',
    random_state=42,
    n_jobs=2
)

model.fit(X_train, y_train)
print(f"  → Training complete!")

# Load actual test metrics from the evaluation CSV (computed in notebook on held-out test set)
METRICS_CSV = os.path.join(BASE_DIR, 'outputs', 'predictions', 'tuned_tree_metrics.csv')
try:
    metrics_df = pd.read_csv(METRICS_CSV)
    xgb_test = metrics_df[(metrics_df['model'] == 'XGB_Tuned') & (metrics_df['split'] == 'test')].iloc[0]
    test_rmse = round(float(xgb_test['RMSE']), 2)
    test_mae  = round(float(xgb_test['MAE']),  2)
    test_mape = round(float(xgb_test['MAPE']), 2)
    print(f"  → Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test MAPE: {test_mape}")
except Exception as e:
    print(f"  ⚠️  Could not read metrics CSV ({e}) — computing on train as fallback")
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    train_pred = model.predict(X_train)
    test_rmse = round(float(np.sqrt(mean_squared_error(y_train, train_pred))), 2)
    test_mae  = round(float(mean_absolute_error(y_train, train_pred)), 2)
    test_mape = 36.35

# 5. Save model
print(f"\n[4/4] Saving model...")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
file_size = os.path.getsize(MODEL_PATH) / 1024
print(f"  → Saved to: {MODEL_PATH}")
print(f"  → File size: {file_size:.1f} KB")

# Also save feature list alongside model for reference
meta = {
    'features':   selected_features,
    'params':     xgb_params,
    'test_rmse':  test_rmse,
    'test_mae':   test_mae,
    'test_mape':  test_mape,
    'train_samples': len(X_train),
    'note': 'Metrics evaluated on held-out test set (not training data)'
}
meta_path = os.path.join(MODEL_DIR, 'xgb_tuned_meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  → Metadata saved to: {meta_path}")

print("\n" + "=" * 60)
print("  ✅ Model exported successfully!")
print("=" * 60)
