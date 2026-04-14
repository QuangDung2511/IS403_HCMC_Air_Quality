# --------------------------------------------------------------------=
# PHASE 1: TINH CHỈNH TRỌNG SỐ MÔ HÌNH THỐNG KÊ (ARIMA Family) - R VERSION
# --------------------------------------------------------------------=
# Mục tiêu: Tìm ra cấu hình tốt nhất cho ARIMA, SARIMA, ARIMAX, SARIMAX.
# Các bước thực hiện:
# 1. Lựa chọn Biến Ngoại sinh: Spearman Correlation & Granger Causality.
# 2. Kiểm tra Tính dừng: Augmented Dickey-Fuller (ADF) Test.
# 3. Auto-ARIMA: Thuật toán Stepwise tối ưu hóa AIC.
# 4. Kiểm định phần dư (Residual Diagnostics): Ljung-Box Test.
# 5. Lưu kết quả: Xuất file .rds phục vụ Ensemble/Hybrid.

# --------------------------------------------------------------------=
# NẠP THƯ VIỆN
# --------------------------------------------------------------------=
library(tidyverse)
library(forecast)
library(tseries) # Dùng cho ADF Test
library(lmtest)  # Dùng cho Granger Causality Test

print("Đã nạp toàn bộ thư viện thành công!")

# --------------------------------------------------------------------=
# HÀM ĐÁNH GIÁ (Regression Metrics - Đảo ngược Logarit)
# --------------------------------------------------------------------=
regression_metrics <- function(y_true, y_pred) {
  y_true_inv <- expm1(y_true)
  y_pred_inv <- expm1(y_pred)
  
  rmse_val <- sqrt(mean((y_true_inv - y_pred_inv)^2))
  mae_val <- mean(abs(y_true_inv - y_pred_inv))
  
  eps <- 1e-6
  mask <- abs(y_true_inv) > eps
  mape_val <- mean(abs((y_true_inv[mask] - y_pred_inv[mask]) / y_true_inv[mask])) * 100
  
  return(c(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val))
}

# --------------------------------------------------------------------=
# LOAD & CHUẨN BỊ DỮ LIỆU
# --------------------------------------------------------------------=
folder_path <- "C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality/data/processed/modeling_fs/"

train_df <- read.csv(paste0(folder_path, "train_dl.csv"))
val_df   <- read.csv(paste0(folder_path, "val_dl.csv"))
test_df  <- read.csv(paste0(folder_path, "test_dl.csv"))

train_y <- train_df$target_pm25_h24_log
val_y   <- val_df$target_pm25_h24_log
test_y  <- test_df$target_pm25_h24_log

ts_train <- ts(train_y, frequency = 24)

cat(sprintf("Train shapes: N=%d\nVal shapes: N=%d\nTest shapes: N=%d\n", 
            nrow(train_df), nrow(val_df), nrow(test_df)))

# --------------------------------------------------------------------=
# 1. EXOGENOUS FEATURE SELECTION (Dành cho ARIMAX & SARIMAX)
# --------------------------------------------------------------------=
print(" BƯỚC 1: Đang lọc biến ngoại sinh...")

# 1.1 Spearman Correlation
all_cols <- colnames(train_df)
feature_cols <- all_cols[!grepl("pm25|datetime_local|target", all_cols)]

spearman_results <- data.frame(Feature = character(), Correlation = numeric())

for (col in feature_cols) {
  corr_val <- cor(train_df[[col]], train_y, method = "spearman")
  spearman_results <- rbind(spearman_results, data.frame(Feature = col, Correlation = abs(corr_val)))
}

# Sắp xếp và lấy Top 5
spearman_results <- spearman_results[order(-spearman_results$Correlation), ]
top_candidates <- head(spearman_results$Feature, 5)

print("=> Top 5 biến (Spearman):")
print(top_candidates)

# 1.2 Granger Causality Test (Thử độ trễ lag = c(1, 6, 12, 24))
selected_exog <- c()
print("--- Kết quả Granger Causality (Tìm p-value < 0.05) ---")

for (col in top_candidates) {
  has_causality <- FALSE
  # Thử nghiệm qua các lag để xem có quan hệ nhân quả không
  for (l in c(1, 6, 12, 24)) {
    tryCatch({
      # grangertest yêu cầu biến (Y ~ X)
      gc_res <- grangertest(train_y ~ train_df[[col]], order = l)
      p_val <- gc_res$`Pr(>F)`[2]
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

print("=> Các biến ngoại sinh chính thức được nạp vào mô hình:")
print(selected_exog)

# Trích xuất ma trận Exogenous
train_exog <- as.matrix(sapply(train_df[, selected_exog], as.numeric))
val_exog   <- as.matrix(sapply(val_df[, selected_exog], as.numeric))
test_exog  <- as.matrix(sapply(test_df[, selected_exog], as.numeric))

# --------------------------------------------------------------------=
# 2. KIỂM ĐỊNH TÍNH DỪNG - Augmented Dickey-Fuller (ADF)
# --------------------------------------------------------------------=
print("\n BƯỚC 2: Kiểm định tính dừng (ADF Test)...")
adf_res <- adf.test(ts_train, alternative = "stationary")
print(adf_res)

if (adf_res$p.value < 0.05) {
  print("=> p-value < 0.05: Bác bỏ H0. Chuỗi LÀ DỪNG (Stationary). Có thể xét d=0.")
} else {
  print("=> p-value >= 0.05: Không thể bác bỏ H0. Chuỗi KHÔNG DỪNG. Cần differencing (d>0).")
}

# --------------------------------------------------------------------=
# 3. AUTO-ARIMA TUNING & DIAGNOSTICS (Stepwise = TRUE để chạy nhanh)
# --------------------------------------------------------------------=
print("\n BƯỚC 3: Dò tìm tham số tự động (Hyperparameter Tuning)...")

# Khởi tạo list để lưu kết quả dự báo
preds_train <- list()
preds_val   <- list()
preds_test  <- list()

# ---------------------------------------------------------
# [1] ARIMA (Non-Seasonal, No Exog)
# ---------------------------------------------------------
print("--- Tuning ARIMA ---")
model_arima <- auto.arima(ts_train, seasonal = FALSE, trace = TRUE, stepwise = TRUE, max.p = 3, max.q = 3)

# Ljung-Box Test (Kiểm tra phần dư có phải nhiễu trắng không)
lb_arima <- Box.test(model_arima$residuals, lag = 24, type = "Ljung-Box")
cat(sprintf("=> Ljung-Box p-value (ARIMA): %f\n", lb_arima$p.value))

# Dự báo trượt (Rolling 1-step Ahead)
test_arima_model <- Arima(test_y, model = model_arima)
preds_test[["ARIMA"]] <- as.numeric(fitted(test_arima_model))


# ---------------------------------------------------------
# [2] ARIMAX (Non-Seasonal, With Exog)
# ---------------------------------------------------------
print("--- Tuning ARIMAX ---")
model_arimax <- auto.arima(ts_train, xreg = train_exog, seasonal = FALSE, trace = TRUE, stepwise = TRUE, max.p = 3, max.q = 3)

test_arimax_model <- Arima(test_y, model = model_arimax, xreg = test_exog)
preds_test[["ARIMAX"]] <- as.numeric(fitted(test_arimax_model))


# ---------------------------------------------------------
# [3] SARIMA (Seasonal m=24, No Exog)
# ---------------------------------------------------------
print("--- Tuning SARIMA ---")
model_sarima <- auto.arima(ts_train, seasonal = TRUE, trace = TRUE, stepwise = TRUE, max.p = 2, max.q = 2, max.P = 1, max.Q = 1)

test_sarima_model <- Arima(test_y, model = model_sarima)
preds_test[["SARIMA"]] <- as.numeric(fitted(test_sarima_model))


# ---------------------------------------------------------
# [4] SARIMAX (Seasonal m=24, With Exog)
# ---------------------------------------------------------
print("--- Tuning SARIMAX ---")
model_sarimax <- auto.arima(ts_train, xreg = train_exog, seasonal = TRUE, trace = TRUE, stepwise = TRUE, max.p = 2, max.q = 2, max.P = 1, max.Q = 1)

test_sarimax_model <- Arima(test_y, model = model_sarimax, xreg = test_exog)
preds_test[["SARIMAX"]] <- as.numeric(fitted(test_sarimax_model))


# --------------------------------------------------------------------=
# 4. LEADERBOARD & LƯU XUẤT 
# --------------------------------------------------------------------=
print("\n --- LEADERBOARD BẢN TUNED TRÊN TEST-SET ---")

res_arima   <- regression_metrics(test_y, preds_test[["ARIMA"]])
res_sarima  <- regression_metrics(test_y, preds_test[["SARIMA"]])
res_arimax  <- regression_metrics(test_y, preds_test[["ARIMAX"]])
res_sarimax <- regression_metrics(test_y, preds_test[["SARIMAX"]])

metric_df <- data.frame(
  Model = c("ARIMA", "SARIMA", "ARIMAX", "SARIMAX"),
  RMSE = c(res_arima["RMSE"], res_sarima["RMSE"], res_arimax["RMSE"], res_sarimax["RMSE"]),
  MAE  = c(res_arima["MAE"], res_sarima["MAE"], res_arimax["MAE"], res_sarimax["MAE"]),
  MAPE = c(res_arima["MAPE"], res_sarima["MAPE"], res_arimax["MAPE"], res_sarimax["MAPE"])
)

metric_df[,-1] <- round(metric_df[,-1], 4)
print(metric_df)

# XUẤT FILE .RDS (Tương đương với xuất Pickle bên Python)
out_dir <- "C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality/outputs/predictions/"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

saveRDS(preds_test, paste0(out_dir, "tuned_stats_preds.rds"))
print(paste("Đã lưu kết quả dự báo 1-Step-Ahead vào:", paste0(out_dir, "tuned_stats_preds.rds")))