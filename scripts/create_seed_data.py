"""
Create Seed Data (72h PM2.5 history)
--------------------------------------
Extracts the last 72 rows from test_split.csv as seed data
so the app has lag features available on first startup.

Run once: python scripts/create_seed_data.py
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'test_split.csv')
SEED_DIR = os.path.join(BASE_DIR, 'data', 'seed')
SEED_PATH = os.path.join(SEED_DIR, 'pm25_history_72h.csv')

print("Loading test data...")
df = pd.read_csv(TEST_PATH)
df['datetime_local'] = pd.to_datetime(df['datetime_local'])

# Take last 72 rows as seed
seed = df[['datetime_local', 'pm25', 'precipitation']].tail(72).copy()
seed.reset_index(drop=True, inplace=True)

os.makedirs(SEED_DIR, exist_ok=True)
seed.to_csv(SEED_PATH, index=False)

print(f"✅ Seed data saved: {len(seed)} rows → {SEED_PATH}")
print(f"   Date range: {seed['datetime_local'].min()} → {seed['datetime_local'].max()}")
print(f"   PM2.5 range: {seed['pm25'].min():.1f} – {seed['pm25'].max():.1f} µg/m³")
