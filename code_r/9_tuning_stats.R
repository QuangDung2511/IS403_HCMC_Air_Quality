# =====================================================================
# BƯỚC 9: TINH CHỈNH TRỌNG SỐ MÔ HÌNH THỐNG KÊ (ARIMA Family)
# =====================================================================
# Mục tiêu: Tìm ra cấu hình tốt nhất cho ARIMA, SARIMA, ARIMAX, SARIMAX.
# 1. Lựa chọn Biến Ngoại sinh: Spearman Correlation & Granger Causality.
# 2. Kiểm tra Tính dừng: Augmented Dickey-Fuller (ADF) Test.
# 3. Auto-ARIMA: Dùng thuật toán Stepwise dựa trên AIC.
# 4. Lưu kết quả: Xuất mảng dự báo ra file .rds

# Nạp thư viện
library(tidyverse)
library(forecast)
library(tseries)
library(lmtest)
library(Metrics)

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN (động, không hard-code)
# ---------------------------------------------------------
# Tự động xác định thư mục script (Hỗ trợ cả RStudio và Rscript CLI)
get_script_path <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, cmd_args)
  if (length(match) > 0) {
    return(dirname(normalizePath(sub(file_arg, "", cmd_args[match]))))
  } else {
    return(dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path)))
  }
}
script_dir <- tryCatch(get_script_path(), error = function(e) getwd())
cat(sprintf("Script directory: %s\n", script_dir))
cat(sprintf("Working directory: %s\n", getwd()))
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

# Đọc từ thư mục R của chính mình (processed_R/modeling_fs/)
in_fs_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
pred_dir  <- file.path(PROJECT_ROOT, "outputs_R/predictions/")

if (!dir.exists(pred_dir)) dir.create(pred_dir, recursive = TRUE)

TARGET_LOG <- "target_pm25_h24_log"

# Hàm đánh giá Metrics (Inverse Log về µg/m³)
regression_metrics <- function(y_true, y_pred) {
  y_true_inv <- expm1(y_true)
  y_pred_inv <- expm1(y_pred)

  rmse_val <- sqrt(mean((y_true_inv - y_pred_inv)^2))
  mae_val  <- mean(abs(y_true_inv - y_pred_inv))

  eps <- 1e-6
  mask <- abs(y_true_inv) > eps
  mape_val <- mean(abs((y_true_inv[mask] - y_pred_inv[mask]) / y_true_inv[mask])) * 100

  return(c(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val))
}

# ---------------------------------------------------------
# ĐỌC DỮ LIỆU
# ---------------------------------------------------------
cat("Đang nạp dữ liệu...\n")
train_df <- read_csv(file.path(in_fs_dir, "train_dl.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
val_df   <- read_csv(file.path(in_fs_dir, "val_dl.csv"),   show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
test_df  <- read_csv(file.path(in_fs_dir, "test_dl.csv"),  show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))

feature_cols <- setdiff(colnames(train_df), c("datetime_local", TARGET_LOG))

X_train <- train_df[, feature_cols]
y_train <- train_df[[TARGET_LOG]]

X_val <- val_df[, feature_cols]
y_val <- val_df[[TARGET_LOG]]

X_test <- test_df[, feature_cols]
y_test <- test_df[[TARGET_LOG]]

# Tạo đối tượng Time-Series cho tập Train với chu kỳ 1 ngày (24 giờ)
ts_train_y <- ts(y_train, frequency = 24)

# =====================================================================
# 1. EXOGENOUS FEATURE SELECTION (Lọc biến ngoại sinh)
# =====================================================================
cat("\n--- 1.1 Spearman Correlation ---\n")
# Tính tương quan Spearman cho các biến không chứa chuỗi "pm25"
exog_candidates <- feature_cols[!grepl("pm25", feature_cols)]
spearman_res <- data.frame(Feature = character(), Spearman_Abs = numeric())

for (col in exog_candidates) {
  corr <- cor(X_train[[col]], y_train, method = "spearman")
  spearman_res <- rbind(spearman_res, data.frame(Feature = col, Spearman_Abs = abs(corr)))
}

spearman_res <- spearman_res %>% arrange(desc(Spearman_Abs))
print(head(spearman_res, 10))

top_candidates <- head(spearman_res$Feature, 5)
cat("\n=> Top 5 ứng viên:", paste(top_candidates, collapse = ", "), "\n")

cat("\n--- 1.2 Granger Causality Test ---\n")
selected_exog <- c()

for (col in top_candidates) {
  has_causality <- FALSE
  cat(sprintf("Thử nghiệm: %s -> PM2.5\n", col))

  # Thử các độ trễ 1, 6, 12, 24 giờ
  for (lag in c(1, 6, 12, 24)) {
    tryCatch({
      # grangertest yêu cầu công thức y ~ x
      gc_test <- grangertest(y_train ~ X_train[[col]], order = lag)
      p_val   <- gc_test$`Pr(>F)`[2] # Lấy p-value của dòng số 2

      if (!is.na(p_val) && p_val < 0.05) {
        has_causality <- TRUE
      }
    }, error = function(e) {})
  }

  if (has_causality) {
    cat(sprintf(" => %s có quan hệ Granger Causality (p < 0.05).\n", col))
    selected_exog <- c(selected_exog, col)
  } else {
    cat(sprintf(" => %s KHÔNG có quan hệ Granger Causality.\n", col))
  }
}

cat("\n=> Các biến ngoại sinh chính thức được chọn:", paste(selected_exog, collapse = ", "), "\n")

# Trích xuất ma trận biến ngoại sinh
train_exog <- as.matrix(X_train[, selected_exog])
val_exog   <- as.matrix(X_val[, selected_exog])
test_exog  <- as.matrix(X_test[, selected_exog])

# =====================================================================
# 2. KIỂM ĐỊNH TÍNH DỪNG (ADF TEST)
# =====================================================================
cat("\n--- 2. ADF Test ---\n")
adf_res <- adf.test(ts_train_y, alternative = "stationary")
print(adf_res)

if (adf_res$p.value < 0.05) {
  cat("p-value < 0.05 => Chuỗi dữ liệu LÀ DỪNG (Stationary). Ta có thể xét d=0.\n")
} else {
  cat("p-value >= 0.05 => Chuỗi dữ liệu LÀ KHÔNG DỪNG (Non-Stationary). Cần d>0.\n")
}

# =====================================================================
# 3. AUTO-ARIMA & RESIDUAL DIAGNOSTICS
# =====================================================================
# ── True 24-Step Walk-Forward Forecast (Leak-Free) ──────────────────────
rolling_forecast_24h <- function(model_fit, all_y, all_exog, start_idx, end_idx) {
  n_steps <- end_idx - start_idx + 1
  preds <- numeric(n_steps)
  
  for (idx in 1:n_steps) {
    i <- start_idx + idx - 1
    y_history <- all_y[1:(i-24)]
    x_history <- if(!is.null(all_exog)) all_exog[1:(i-24), , drop=FALSE] else NULL
    x_future  <- if(!is.null(all_exog)) matrix(rep(all_exog[i-24, ], 24), nrow=24, byrow=TRUE) else NULL
    
    if (!is.null(x_history)) {
      temp_fit <- Arima(y_history, model = model_fit, xreg = x_history)
      fc <- forecast(temp_fit, h = 24, xreg = x_future)
    } else {
      temp_fit <- Arima(y_history, model = model_fit)
      fc <- forecast(temp_fit, h = 24)
    }
    preds[idx] <- as.numeric(tail(fc$mean, 1))
    
    if (idx %% 500 == 0) cat(sprintf("    Progress: %d/%d\n", idx, n_steps))
  }
  return(preds)
}

# Chuẩn bị dữ liệu Global
all_y_global <- c(y_train, y_val, y_test)
all_exog_global <- if (length(selected_exog) > 0) rbind(train_exog, val_exog, test_exog) else NULL

val_start_idx  <- length(y_train) + 1
val_end_idx    <- length(y_train) + length(y_val)
test_start_idx <- val_end_idx + 1
test_end_idx   <- val_end_idx + length(y_test)

predictions_dict <- list()

# --- 3.1 ARIMA (Non-Seasonal, No Exog) ---
cat("\n--- Tuning ARIMA ---\n")
model_arima <- auto.arima(ts_train_y, seasonal = FALSE, trace = TRUE, stepwise = TRUE, max.p = 3, max.q = 3)

# Ljung-Box Test cho ARIMA
lb_arima <- Box.test(model_arima$residuals, lag = 24, type = "Ljung-Box")
cat(sprintf("Ljung-Box p-value (ARIMA): %f\n", lb_arima$p.value))

# Dự báo trượt (Rolling 24-step ahead) trên Val và Test
predictions_dict[["ARIMA_train"]] <- as.numeric(fitted(model_arima))

cat("  Rolling forecast ARIMA on Val set...\n")
predictions_dict[["ARIMA_val"]]  <- rolling_forecast_24h(model_arima, all_y_global, NULL, val_start_idx, val_end_idx)

cat("  Rolling forecast ARIMA on Test set...\n")
predictions_dict[["ARIMA_test"]] <- rolling_forecast_24h(model_arima, all_y_global, NULL, test_start_idx, test_end_idx)

# Sanity check: Độ tương quan với giá trị thực bước trước (nên thấp nếu không leak)
lag_corr_arima <- cor(predictions_dict[["ARIMA_test"]][2:length(y_test)], y_test[1:(length(y_test)-1)])
cat(sprintf("Sanity Correlation ARIMA: %.4f\n", lag_corr_arima))

# --- 3.2 ARIMAX (Non-Seasonal, With Exog) ---
cat("\n--- Tuning ARIMAX ---\n")
model_arimax <- auto.arima(ts_train_y, xreg = train_exog, seasonal = FALSE, trace = TRUE, stepwise = TRUE, max.p = 3, max.q = 3)

predictions_dict[["ARIMAX_train"]] <- as.numeric(fitted(model_arimax))

cat("  Rolling forecast ARIMAX on Val set...\n")
predictions_dict[["ARIMAX_val"]]  <- rolling_forecast_24h(model_arimax, all_y_global, all_exog_global, val_start_idx, val_end_idx)

cat("  Rolling forecast ARIMAX on Test set...\n")
predictions_dict[["ARIMAX_test"]] <- rolling_forecast_24h(model_arimax, all_y_global, all_exog_global, test_start_idx, test_end_idx)

lag_corr_arimax <- cor(predictions_dict[["ARIMAX_test"]][2:length(y_test)], y_test[1:(length(y_test)-1)])
cat(sprintf("Sanity Correlation ARIMAX: %.4f\n", lag_corr_arimax))

# --- 3.3 SARIMA (Seasonal m=24, No Exog) ---
cat("\n--- Tuning SARIMA ---\n")
model_sarima <- auto.arima(ts_train_y, seasonal = TRUE, trace = TRUE, stepwise = TRUE, max.p = 2, max.q = 2, max.P = 1, max.Q = 1)

predictions_dict[["SARIMA_train"]] <- as.numeric(fitted(model_sarima))

cat("  Rolling forecast SARIMA on Val set...\n")
predictions_dict[["SARIMA_val"]]  <- rolling_forecast_24h(model_sarima, all_y_global, NULL, val_start_idx, val_end_idx)

cat("  Rolling forecast SARIMA on Test set...\n")
predictions_dict[["SARIMA_test"]] <- rolling_forecast_24h(model_sarima, all_y_global, NULL, test_start_idx, test_end_idx)

lag_corr_sarima <- cor(predictions_dict[["SARIMA_test"]][2:length(y_test)], y_test[1:(length(y_test)-1)])
cat(sprintf("Sanity Correlation SARIMA: %.4f\n", lag_corr_sarima))

# --- 3.4 SARIMAX (Seasonal m=24, With Exog) ---
cat("\n--- Tuning SARIMAX ---\n")
model_sarimax <- auto.arima(ts_train_y, xreg = train_exog, seasonal = TRUE, trace = TRUE, stepwise = TRUE, max.p = 2, max.q = 2, max.P = 1, max.Q = 1)

predictions_dict[["SARIMAX_train"]] <- as.numeric(fitted(model_sarimax))

cat("  Rolling forecast SARIMAX on Val set...\n")
predictions_dict[["SARIMAX_val"]]  <- rolling_forecast_24h(model_sarimax, all_y_global, all_exog_global, val_start_idx, val_end_idx)

cat("  Rolling forecast SARIMAX on Test set...\n")
predictions_dict[["SARIMAX_test"]] <- rolling_forecast_24h(model_sarimax, all_y_global, all_exog_global, test_start_idx, test_end_idx)

lag_corr_sarimax <- cor(predictions_dict[["SARIMAX_test"]][2:length(y_test)], y_test[1:(length(y_test)-1)])
cat(sprintf("Sanity Correlation SARIMAX: %.4f\n", lag_corr_sarimax))

# =====================================================================
# 4. LEADERBOARD NHÓM THỐNG KÊ & XUẤT FILE
# =====================================================================
cat("\n --- LEADERBOARD BẢN TUNED TRÊN TEST-SET ---\n")

model_names <- c("ARIMA", "ARIMAX", "SARIMA", "SARIMAX")
results <- data.frame()

for (m in model_names) {
  res_te <- regression_metrics(y_test, predictions_dict[[paste0(m, "_test")]])
  results <- rbind(results, data.frame(Model = m, RMSE = res_te["RMSE"], MAE = res_te["MAE"], MAPE = res_te["MAPE"]))
}

results[,-1] <- round(results[,-1], 4)
print(results)

# Đóng gói và lưu mảng dự báo ra định dạng .rds (Tương đương .pkl của Python)
saveRDS(predictions_dict, file.path(pred_dir, "tuned_stats_preds.rds"))

# Xuất thêm bản Real Units (µg/m³) để đối chiếu trực tiếp với các model khác
predictions_dict_real <- lapply(predictions_dict, expm1)
saveRDS(predictions_dict_real, file.path(pred_dir, "tuned_stats_preds_real_units.rds"))

cat("\n Đã lưu kết quả dự báo (Log-space) tại:", file.path(pred_dir, "tuned_stats_preds.rds"))
cat("\n Đã lưu kết quả dự báo (Real-unit) tại:", file.path(pred_dir, "tuned_stats_preds_real_units.rds\n"))