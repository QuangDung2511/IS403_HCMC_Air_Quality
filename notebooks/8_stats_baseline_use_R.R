# --------------------------------------------------------------------=
# BƯỚC 8: MÔ HÌNH THỐNG KÊ VÀ HỌC SÂU - DỰ BÁO PM2.5 (t + 24h)
# --------------------------------------------------------------------=
# 1. Các mô hình Thống kê: ARIMA, SARIMA, ARIMAX, SARIMAX
# - Biến nội sinh (endog): target_pm25_h24_log
# - Biến ngoại sinh (exog): Chọn top 5 đặc trưng mạnh nhất (thời tiết).
# Sử dụng cơ chế 1-Step Ahead Rolling Forecast trên tập Test giống Python.

# --------------------------------------------------------------------=
# NẠP THƯ VIỆN
# --------------------------------------------------------------------=
library(tidyverse)
library(forecast)

print("Đã nạp thư viện thành công!")

# --------------------------------------------------------------------=
# HÀM ĐÁNH GIÁ (Regression Metrics)
# --------------------------------------------------------------------=
regression_metrics <- function(y_true, y_pred) {
  # Hoàn tác logarit (inverse log) để tính toán sai số bằng µg/m³
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
# Nhớ thay đổi đường dẫn nếu cần thiết
folder_path <- "C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality/data/processed/modeling_fs/"

train_df <- read.csv(paste0(folder_path, "train_dl.csv"))
val_df   <- read.csv(paste0(folder_path, "val_dl.csv"))
test_df  <- read.csv(paste0(folder_path, "test_dl.csv"))

# Lấy biến Target
train_y <- train_df$target_pm25_h24_log
test_y  <- test_df$target_pm25_h24_log

ts_train <- ts(train_y, frequency = 24)

# Lấy 5 biến ngoại sinh (Exogenous variables) đầu tiên
all_cols <- colnames(train_df)
exog_candidates <- all_cols[!grepl("pm25|datetime_local|target", all_cols)]
exog_cols <- head(exog_candidates, 5)

print("Các biến ngoại sinh (Exogenous variables) được chọn:")
print(exog_cols)

# Ép chặt kiểu ma trận
train_exog <- as.matrix(sapply(train_df[, exog_cols], as.numeric))
test_exog  <- as.matrix(sapply(test_df[, exog_cols], as.numeric))
colnames(test_exog) <- colnames(train_exog)

# --------------------------------------------------------------------=
# HUẤN LUYỆN VÀ DỰ BÁO CÁC MÔ HÌNH THỐNG KÊ
# --------------------------------------------------------------------=
print("Bắt đầu huấn luyện và áp dụng mô hình (1-Step Ahead)...")

# ---------------------------------------------------------
# Chạy ARIMA: order=(1,1,1)
# ---------------------------------------------------------
print("1/4. ARIMA...")
model_arima <- Arima(ts_train, order = c(1, 1, 1))
test_arima  <- Arima(test_y, model = model_arima)
pred_arima  <- as.numeric(fitted(test_arima))

# ---------------------------------------------------------
# Chạy SARIMA: order=(1,0,1), seasonal=(1,0,0,24)
# ---------------------------------------------------------
print("2/4. SARIMA...")
model_sarima <- Arima(ts_train, order = c(1, 0, 1), seasonal = list(order = c(1, 0, 0), period = 24))
test_sarima  <- Arima(test_y, model = model_sarima)
pred_sarima  <- as.numeric(fitted(test_sarima))

# ---------------------------------------------------------
# Chạy ARIMAX: order=(1,1,1), dùng Exogenous variables
# ---------------------------------------------------------
print("3/4. ARIMAX...")
model_arimax <- Arima(ts_train, order = c(1, 1, 1), xreg = train_exog)
test_arimax  <- Arima(test_y, model = model_arimax, xreg = test_exog)
pred_arimax  <- as.numeric(fitted(test_arimax))

# ---------------------------------------------------------
# Chạy SARIMAX: order=(1,0,1), seasonal=(1,0,0,24), dùng Exogenous
# ---------------------------------------------------------
print("4/4. SARIMAX...")
model_sarimax <- Arima(ts_train, order = c(1, 0, 1), seasonal = list(order = c(1, 0, 0), period = 24), xreg = train_exog)
test_sarimax  <- Arima(test_y, model = model_sarimax, xreg = test_exog)
pred_sarimax  <- as.numeric(fitted(test_sarimax))

print("Hoàn tất dự báo!")

# --------------------------------------------------------------------=
# TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (TEST METRICS)
# --------------------------------------------------------------------=
res_arima   <- regression_metrics(test_y, pred_arima)
res_sarima  <- regression_metrics(test_y, pred_sarima)
res_arimax  <- regression_metrics(test_y, pred_arimax)
res_sarimax <- regression_metrics(test_y, pred_sarimax)

results_df <- data.frame(
  Model = c("ARIMA", "SARIMA", "ARIMAX", "SARIMAX"),
  RMSE = c(res_arima["RMSE"], res_sarima["RMSE"], res_arimax["RMSE"], res_sarimax["RMSE"]),
  MAE  = c(res_arima["MAE"], res_sarima["MAE"], res_arimax["MAE"], res_sarimax["MAE"]),
  MAPE = c(res_arima["MAPE"], res_sarima["MAPE"], res_arimax["MAPE"], res_sarimax["MAPE"])
)

results_df[,-1] <- round(results_df[,-1], 4)

print("\n --= Bảng metrics trên tập Test (Thống kê) --=")
print(results_df)

# --------------------------------------------------------------------=
# 6. TRỰC QUAN HÓA (Actual vs Predicted - 200 giờ đầu tiên của tập Test)
# --------------------------------------------------------------------=
print("Đang vẽ biểu đồ...")

# Chọn 200 giờ đầu tiên
subset_size <- min(200, length(test_y))

# Đảo ngược logarit (expm1) để vẽ đồ thị ở thang đo thực tế (µg/m³)
actual_inv   <- expm1(test_y)[1:subset_size]
arima_inv    <- expm1(pred_arima)[1:subset_size]
sarima_inv   <- expm1(pred_sarima)[1:subset_size]
arimax_inv   <- expm1(pred_arimax)[1:subset_size]
sarimax_inv  <- expm1(pred_sarimax)[1:subset_size]

# Vẽ biểu đồ đường
plot(1:subset_size, actual_inv, type = "l", col = "black", lwd = 3,
     main = "Actual vs Predicted theo thời gian (200 giờ đầu của tập Test)",
     ylab = "Nồng độ PM2.5 (µg/m³)", xlab = "Thời gian (Giờ)")

# Thêm các đường dự báo của 4 mô hình
lines(1:subset_size, arima_inv,   col = "orange", lty = 2, lwd = 2)
lines(1:subset_size, sarima_inv,  col = "green",  lty = 2, lwd = 2)
lines(1:subset_size, arimax_inv,  col = "purple", lty = 3, lwd = 2)
lines(1:subset_size, sarimax_inv, col = "blue",   lty = 3, lwd = 2)

# Thêm chú thích (Legend)
legend("topleft", legend = c("Actual PM2.5 (t+24h)", "ARIMA", "SARIMA", "ARIMAX", "SARIMAX"), 
       col = c("black", "orange", "green", "purple", "blue"), 
       lty = c(1, 2, 2, 3, 3), lwd = c(3, 2, 2, 2, 2))

print("Vẽ biểu đồ thành công!")