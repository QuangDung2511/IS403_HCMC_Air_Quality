# =====================================================================
# BƯỚC 7: SO SÁNH MÔ HÌNH TREE-BASED - DỰ BÁO PM2.5 (t + 24h)
# =====================================================================

# Nạp thư viện
library(tidyverse)
library(randomForest)
library(xgboost)
library(lightgbm)
library(ggplot2)

# Cấu hình giao diện biểu đồ
theme_set(theme_minimal(base_size = 14))
RANDOM_STATE <- 42

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN (động, không hard-code)
# ---------------------------------------------------------
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

# Đọc file từ thư mục R của chính mình (processed_R/modeling_fs/)
in_fs_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")

# Lưu output vào thư mục riêng của R
fig_dir <- file.path(PROJECT_ROOT, "outputs_R/figures/")
out_csv <- file.path(PROJECT_ROOT, "outputs_R/predictions/tree_model_metrics.csv")

if (!dir.exists(fig_dir))          dir.create(fig_dir,          recursive = TRUE)
if (!dir.exists(dirname(out_csv))) dir.create(dirname(out_csv), recursive = TRUE)

TARGET <- "target_pm25_h24"

# ---------------------------------------------------------
# HÀM ĐÁNH GIÁ METRICS (Không dùng expm1 vì target chưa log)
# ---------------------------------------------------------
regression_metrics <- function(y_true, y_pred) {
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae  <- mean(abs(y_true - y_pred))
  eps  <- 1e-6
  mask <- abs(y_true) > eps
  mape <- mean(abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
  return(c(RMSE = rmse, MAE = mae, MAPE = mape))
}

# ---------------------------------------------------------
# 1. ĐỌC DỮ LIỆU
# ---------------------------------------------------------
cat("Đang nạp dữ liệu...\n")
train_df <- read_csv(file.path(in_fs_dir, "train_tree.csv"), show_col_types = FALSE)
val_df   <- read_csv(file.path(in_fs_dir, "val_tree.csv"),   show_col_types = FALSE)
test_df  <- read_csv(file.path(in_fs_dir, "test_tree.csv"),  show_col_types = FALSE)

# Loại bỏ NA ở target
train_df <- train_df %>% drop_na(all_of(TARGET))
val_df   <- val_df   %>% drop_na(all_of(TARGET))
test_df  <- test_df  %>% drop_na(all_of(TARGET))

feature_cols <- setdiff(colnames(train_df), c("datetime_local", TARGET))
cat(sprintf("Train: %d dòng | Val: %d dòng | Test: %d dòng\n", nrow(train_df), nrow(val_df), nrow(test_df)))

# Trích xuất ma trận cho XGBoost & LightGBM
X_train <- as.matrix(train_df[, feature_cols])
y_train <- train_df[[TARGET]]

X_val <- as.matrix(val_df[, feature_cols])
y_val <- val_df[[TARGET]]

X_test <- as.matrix(test_df[, feature_cols])
y_test <- test_df[[TARGET]]

# ---------------------------------------------------------
# 2. HUẤN LUYỆN MÔ HÌNH (TRAINING)
# ---------------------------------------------------------
preds_train <- list()
preds_val   <- list()
preds_test  <- list()

# --- 2.1 Random Forest ---
cat("Đang huấn luyện RandomForest...\n")
set.seed(RANDOM_STATE)
rf_model <- randomForest(x = X_train, y = y_train, ntree = 200, nodesize = 2)
preds_train[["RandomForest"]] <- predict(rf_model, X_train)
preds_val[["RandomForest"]]   <- predict(rf_model, X_val)
preds_test[["RandomForest"]]  <- predict(rf_model, X_test)
cat("✅ RandomForest fit xong\n")

# --- 2.2 XGBoost ---
cat("Đang huấn luyện XGBoost...\n")
dtrain_xgb <- xgb.DMatrix(data = X_train, label = y_train)
xgb_params <- list(max_depth = 6, eta = 0.08, subsample = 0.9, colsample_bytree = 0.9,
                   tree_method = "hist", objective = "reg:squarederror", seed = RANDOM_STATE)
xgb_model <- xgb.train(params = xgb_params, data = dtrain_xgb, nrounds = 200)
preds_train[["XGBoost"]] <- predict(xgb_model, X_train)
preds_val[["XGBoost"]]   <- predict(xgb_model, X_val)
preds_test[["XGBoost"]]  <- predict(xgb_model, X_test)
cat("✅ XGBoost fit xong\n")

# --- 2.3 LightGBM ---
cat("Đang huấn luyện LightGBM...\n")
dtrain_lgb <- lgb.Dataset(data = X_train, label = y_train)
lgb_params <- list(objective = "regression", metric = "rmse", max_depth = 8, num_leaves = 63,
                   learning_rate = 0.05, subsample = 0.9, colsample_bytree = 0.9, seed = RANDOM_STATE)
lgb_model <- lgb.train(params = lgb_params, data = dtrain_lgb, nrounds = 200, verbose = -1)
preds_train[["LightGBM"]] <- predict(lgb_model, X_train)
preds_val[["LightGBM"]]   <- predict(lgb_model, X_val)
preds_test[["LightGBM"]]  <- predict(lgb_model, X_test)
cat("✅ LightGBM fit xong\n")

# ---------------------------------------------------------
# 3. TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (EVALUATION)
# ---------------------------------------------------------
model_names <- c("RandomForest", "XGBoost", "LightGBM")
results <- data.frame()

for (m in model_names) {
  res_tr <- regression_metrics(y_train, preds_train[[m]])
  res_va <- regression_metrics(y_val,   preds_val[[m]])
  res_te <- regression_metrics(y_test,  preds_test[[m]])

  results <- rbind(results,
                   data.frame(model = m, split = "train", RMSE = res_tr[1], MAE = res_tr[2], MAPE = res_tr[3]),
                   data.frame(model = m, split = "val",   RMSE = res_va[1], MAE = res_va[2], MAPE = res_va[3]),
                   data.frame(model = m, split = "test",  RMSE = res_te[1], MAE = res_te[2], MAPE = res_te[3]))
}

# Lưu CSV
write_csv(results, out_csv)

# Hiển thị Leaderboard (Test Set)
cat("\n🎯 --- LEADERBOARD TRÊN TẬP TEST ---\n")
test_metrics <- results %>% filter(split == "test") %>% mutate(across(where(is.numeric), ~round(., 4)))
print(test_metrics)

# ---------------------------------------------------------
# 4. TRỰC QUAN HÓA (VISUALIZATION)
# ---------------------------------------------------------
cat("\nĐang vẽ và lưu biểu đồ...\n")

# 4.1 Biểu đồ cột: RMSE / MAE / MAPE
plot_df <- test_metrics %>% pivot_longer(cols = c(RMSE, MAE, MAPE), names_to = "metric", values_to = "value")
p_bar <- ggplot(plot_df, aes(x = metric, y = value, fill = model)) +
  geom_col(position = "dodge", color = "black", alpha = 0.8) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "So sánh metric trên tập Test (PM2.5 t+24h)", x = "Metric", y = "Giá trị")

ggsave(file.path(fig_dir, "tree_models_test_metrics_bar.png"), plot = p_bar, width = 10, height = 5)
print(p_bar)

# 4.2 Scatter: Actual vs Predicted (1x3 Subplots dùng Facet)
scatter_df <- data.frame(Actual = y_test, RandomForest = preds_test[["RandomForest"]],
                         XGBoost = preds_test[["XGBoost"]], LightGBM = preds_test[["LightGBM"]]) %>%
  pivot_longer(cols = -Actual, names_to = "Model", values_to = "Predicted") %>%
  mutate(Model = factor(Model, levels = model_names))

lim_val <- max(y_test) * 1.05

p_scatter <- ggplot(scatter_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.25, color = "steelblue", size = 1.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  coord_cartesian(xlim = c(0, lim_val), ylim = c(0, lim_val)) +
  facet_wrap(~ Model) +
  labs(title = "Test set: Actual vs Predicted (µg/m³)")

ggsave(file.path(fig_dir, "tree_models_test_actual_vs_predicted.png"), plot = p_scatter, width = 14, height = 4.5)
print(p_scatter)

# 4.3 Chuỗi thời gian (Time-series) 200 giờ đầu
k <- min(200, length(y_test))
ts_df <- data.frame(Hour = 1:k, Actual = y_test[1:k],
                    RandomForest = preds_test[["RandomForest"]][1:k],
                    XGBoost = preds_test[["XGBoost"]][1:k],
                    LightGBM = preds_test[["LightGBM"]][1:k]) %>%
  pivot_longer(cols = -Hour, names_to = "Legend", values_to = "PM2.5")

p_ts <- ggplot(ts_df, aes(x = Hour, y = PM2.5, color = Legend, linetype = Legend, size = Legend)) +
  geom_line(alpha = 0.8) +
  scale_color_manual(values    = c("Actual" = "black",  "RandomForest" = "red",    "XGBoost" = "blue", "LightGBM" = "green")) +
  scale_linetype_manual(values = c("Actual" = "solid",  "RandomForest" = "dashed", "XGBoost" = "dotted", "LightGBM" = "dotdash")) +
  scale_size_manual(values     = c("Actual" = 1.2,      "RandomForest" = 0.8,      "XGBoost" = 0.8,    "LightGBM" = 0.8)) +
  labs(title = "Actual vs Predicted theo thời gian (200 giờ đầu tập Test)", x = "Giờ", y = "PM2.5 (µg/m³)")

ggsave(file.path(fig_dir, "tree_models_test_timeseries_subset.png"), plot = p_ts, width = 12, height = 4)
print(p_ts)

cat(" Hoàn tất toàn bộ quy trình! Các biểu đồ và file CSV đã được lưu.\n")