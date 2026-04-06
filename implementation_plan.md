# Final Consolidated Pipeline Architecture

This document serves as the master blueprint for your machine learning pipeline. It resolves both the **Data-Leakage** issues and the **Logical EDA Flow** dilemma specific to time-series forecasting.

## The Strategy

We must balance three conflicting rules:
1. **No Data Leakage**: We cannot calculate global math (like `mean` or `StandardScaler`) using the test set.
2. **Time-Series Lags**: We *must* calculate lag features (like `rolling_mean(24)`) on a continuous timeline before splitting, otherwise we break the time-window for the first validation rows.
3. **Exploratory Flow**: We must do EDA on base features *early* to get insights, but we can't look at future dates.

**The Solution:** We load the continuous dataset into the EDA notebook, but artificially slice it to the first 60% of dates (the Train window). Thus, we freely explore base features without snooping into the future. Then, we use the continuous dataset to build lag features, and only then do we officially split the dataset for stateful math.

---

## 1. Project Implementation Plan

### Phase 1: Foundation (Continuous Data)

#### [MODIFY] `1_basic_cleaning.ipynb`
*   **Action**: Merges raw data, interpolates gaps, sets timestamps.
*   **Output**: `hcmc_merged_cleaned.csv` (The 14 "Base" Features).

#### [MODIFY] `2_EDA_train_only.ipynb` (Newly rearranged)
*   **Action**: Loads `hcmc_merged_cleaned.csv` but **immediately truncates it** to the top 60% of rows (`df = df.iloc[:int(len(df)*0.6)]`).
*   **Purpose**: Perform classic EDA (distributions, boxplots, correlation matrices) purely on the *historical training view* of the Base Features. No overwhelming 48 engineered lags, and no test-set leakage.

#### [MODIFY] `3_feature_engineering_stateless.ipynb` (Newly rearranged)
*   **Action**: Loads the *full* continuous `hcmc_merged_cleaned.csv` (100% of rows).
*   **Purpose**: Calculates "Stateless" features. Applies `shift()` for lags, rolling averages, and creates the target 24-hr variable. Doing this continuously prevents messy cross-dataset border math. 
*   **Clean Up**: Drops NA rows created at the start by lags.
*   **Output**: `hcmc_features_continuous.csv` (The 48+ highly engineered features).

---

### Phase 2: Official Splitting & Stateful Preprocessing

#### [MODIFY] `4_train_val_test_split.ipynb`
*   **Action**: Loads the highly engineered `hcmc_features_continuous.csv`.
*   **Purpose**: Performs the strict 60 / 20 / 20 chronological split. Physical datasets are formally separated from here on out.
*   **Output**: `train_split.csv`, `val_split.csv`, `test_split.csv`.

#### [MODIFY] `5_base_preprocessing_and_selection.ipynb`
*   **Action**: Loads the isolated split datasets.
*   **Purpose**: Handles "Stateful" variables.
    *   Fits `LabelEncoder` strictly on `Train`, transforms everything.
    *   Trains Random Forest / XGBoost strictly on `Train` to compute Feature Importance.
    *   Trims the 48+ features down to Top-K robust predictors.
*   **Output**: Cleaned tree-modeling arrays.

#### [MODIFY] `6_preprocessing_dl.ipynb`
*   **Action**: Loads tree-modeling arrays.
*   **Purpose**: Applies heavy stateful math. Fits `StandardScaler` strictly on `Train`, transforms all.
*   **Output**: Normalized deep-learning arrays.

#### [MODIFY] `7_train_eval_tree_models.ipynb`
*   **Action**: Trains final baseline models and scores metrics (RMSE, MAPE).

---

## User Review Required

Does this consolidated blueprint make complete sense?

By utilizing this architecture, your Jupyter workspace perfectly mirrors industry standards for production-grade Time Series Forecasting. If you approve this plan, I will immediately run scripts to physically rename notebooks 2, 3, and 4 and inject the `int(len(df)*0.6)` slicing logic into the EDA notebook!
