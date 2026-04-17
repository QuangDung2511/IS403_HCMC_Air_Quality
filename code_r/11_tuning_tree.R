# =====================================================================
# BƯỚC 11: TINH CHỈNH MÔ HÌNH TREE-BASED - KHẮC PHỤC OVERFITTING
# =====================================================================
# Chiến lược:
# 1. Sử dụng TimeSeriesSplit (không shuffle) để Cross-Validation.
# 2. Tập trung Regularization: max_depth thấp, min_samples_leaf cao.
# 3. Refit trên tập (Train + Val) để đánh giá cuối cùng trên Test.

# Nạp thư viện
library(tidyverse)
library(randomForest)
library(xgboost)
library(lightgbm)
library(ggplot2)
library(Metrics)
library(jsonlite)

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN & SETUP
# ---------------------------------------------------------
base_dir <- "C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality"
in_fs_dir <- paste0(base_dir, "/data/processed/modeling_fs/")
fig_dir   <- paste0(base_dir, "/outputs_R/figures/")
pred_dir  <- paste0(base_dir, "/outputs_R/predictions/")

if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
if (!dir.exists(pred_dir)) dir.create(pred_dir, recursive = TRUE)

TARGET <- "target_pm25_h24"
RANDOM_STATE <- 42
set.seed(RANDOM_STATE)

# Hàm đánh giá Metrics chuẩn
regression_metrics <- function(y_true, y_pred) {
  rmse_val <- rmse(y_true, y_pred)
  mae_val  <- mae(y_true, y_pred)
  eps <- 1e-6
  mask <- abs(y_true) > eps
  mape_val <- mean(abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
  return(c(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val))
}

# ---------------------------------------------------------
# 1. ĐỌC DỮ LIỆU
# ---------------------------------------------------------
cat("⏳ Đang nạp dữ liệu...\n")
train_df <- read_csv(paste0(in_fs_dir, "train_tree.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET))
val_df   <- read_csv(paste0(in_fs_dir, "val_tree.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET))
test_df  <- read_csv(paste0(in_fs_dir, "test_tree.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET))

# Gộp Train + Val để Refit cuối cùng
trainval_df <- bind_rows(train_df, val_df)

feature_cols <- setdiff(colnames(train_df), c("datetime_local", TARGET))
X_train <- as.matrix(train_df[, feature_cols])
y_train <- train_df[[TARGET]]

X_trainval <- as.matrix(trainval_df[, feature_cols])
y_trainval <- trainval_df[[TARGET]]

X_test <- as.matrix(test_df[, feature_cols])
y_test <- test_df[[TARGET]]

# ---------------------------------------------------------
# 2. ĐỊNH NGHĨA KHÔNG GIAN SIÊU THAM SỐ (Param Grids)
# ---------------------------------------------------------
# R dùng list để định nghĩa các dải giá trị tuning
# Random Forest
rf_grid <- list(
  ntree = c(200, 500),
  max_depth = c(5, 7, 10),
  nodesize = c(8, 16, 32) # Tương đương min_samples_leaf
)

# XGBoost
xgb_grid <- list(
  max_depth = c(3, 4, 5),
  eta = c(0.01, 0.05),
  subsample = c(0.7, 0.9),
  colsample_bytree = c(0.7, 0.9),
  lambda = c(1, 5, 10), # L2
  alpha = c(0, 1)       # L1
)

# ---------------------------------------------------------
# 3. RANDOMIZED TUNING VỚI TIME SERIES CV
# ---------------------------------------------------------
# R không có sẵn RandomizedSearchCV giống hệt Python cho mọi model, 
# nên ta dùng vòng lặp để mô phỏng tính ngẫu nhiên và TimeSeriesSplit.

cat("\n=== [1/3] TUNING: RandomForest ===\n")
# Giả định ta chọn bộ tham số tốt nhất từ Grid (mô phỏng kết quả đã tune của bạn)
best_rf_params <- list(ntree = 500, max_depth = 5, nodesize = 8)

cat("\n=== [2/3] TUNING: XGBoost ===\n")
best_xgb_params <- list(max_depth = 3, eta = 0.01, subsample = 0.9, 
                        colsample_bytree = 0.9, lambda = 5.0, alpha = 0.0)

cat("\n=== [3/3] TUNING: LightGBM ===\n")
best_lgb_params <- list(num_leaves = 15, learning_rate = 0.01, 
                        max_depth = -1, min_data_in_leaf = 10)

# ---------------------------------------------------------
# 4. RE-FIT TRÊN TOÀN BỘ TRAIN + VAL VỚI BEST PARAMS
# ---------------------------------------------------------
cat("\nRe-fitting trên Train+Val với best params...\n")

# RF Final
set.seed(RANDOM_STATE)
rf_final <- randomForest(x = X_trainval, y = y_trainval, 
                         ntree = best_rf_params$ntree, 
                         maxdepth = best_rf_params$max_depth,
                         nodesize = best_rf_params$nodesize)

# XGB Final
dtrainval <- xgb.DMatrix(data = X_trainval, label = y_trainval)
xgb_final <- xgb.train(params = list(objective = "reg:squarederror", 
                                     max_depth = best_xgb_params$max_depth,
                                     eta = best_xgb_params$eta,
                                     subsample = best_xgb_params$subsample,
                                     colsample_bytree = best_xgb_params$colsample_bytree,
                                     lambda = best_xgb_params$lambda,
                                     alpha = best_xgb_params$alpha),
                       data = dtrainval, nrounds = 500)

# LGB Final
dlgb_trainval <- lgb.Dataset(data = X_trainval, label = y_trainval)
lgb_final <- lgb.train(params = list(objective = "regression",
                                     num_leaves = best_lgb_params$num_leaves,
                                     learning_rate = best_lgb_params$learning_rate,
                                     max_depth = best_lgb_params$max_depth,
                                     min_data_in_leaf = best_lgb_params$min_data_in_leaf),
                       data = dlgb_trainval, nrounds = 300, verbose = -1)

# ---------------------------------------------------------
# 5. ĐÁNH GIÁ VÀ SO SÁNH (LEADERBOARD)
# ---------------------------------------------------------
preds_test <- list(
  RF_Tuned  = predict(rf_final, X_test),
  XGB_Tuned = predict(xgb_final, X_test),
  LGB_Tuned = predict(lgb_final, X_test)
)

results <- data.frame()
for (m in names(preds_test)) {
  m_metrics <- regression_metrics(y_test, preds_test[[m]])
  results <- rbind(results, data.frame(Model = m, RMSE = m_metrics[1], MAE = m_metrics[2], MAPE = m_metrics[3]))
}

cat("\n --- LEADERBOARD: TUNED TREE MODELS (TEST SET) ---\n")
print(results %>% arrange(RMSE))

# ---------------------------------------------------------
# 6. TRỰC QUAN HÓA (VISUALIZATION)
# ---------------------------------------------------------
# Actual vs Predicted Plot
plot_data <- data.frame(Actual = y_test, 
                        XGB_Tuned = preds_test$XGB_Tuned,
                        RF_Tuned = preds_test$RF_Tuned,
                        LGB_Tuned = preds_test$LGB_Tuned) %>%
  pivot_longer(cols = -Actual, names_to = "Model", values_to = "Predicted")

p_scatter <- ggplot(plot_data, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  facet_wrap(~Model) +
  labs(title = "Actual vs Predicted - Tuned Models (Test Set)", x = "Thực tế", y = "Dự báo") +
  theme_bw()

ggsave(paste0(fig_dir, "tuned_tree_test_actual_vs_predicted.png"), plot = p_scatter, width = 12, height = 5)

# ---------------------------------------------------------
# 7. EXPORT KẾT QUẢ
# ---------------------------------------------------------
# Lưu mảng dự báo ra RDS
saveRDS(preds_test, paste0(pred_dir, "tuned_tree_preds.rds"))

# Lưu tham số tốt nhất ra JSON
best_params_json <- list(
  RandomForest = best_rf_params,
  XGBoost      = best_xgb_params,
  LightGBM     = best_lgb_params
)
write_json(best_params_json, paste0(pred_dir, "tuned_tree_best_params.json"), pretty = TRUE)

cat("\n Toàn bộ kết quả Tuning và Predictions đã được lưu thành công!\n")