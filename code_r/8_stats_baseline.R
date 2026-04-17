# =====================================================================
# BƯỚC 8: MÔ HÌNH THỐNG KÊ VÀ HỌC SÂU - DỰ BÁO PM2.5 (t + 24h)
# =====================================================================
# Mục tiêu: Huấn luyện ARIMA, SARIMA, ARIMAX, SARIMAX và LSTM, GRU.
# Đầu vào: train_dl.csv, val_dl.csv, test_dl.csv (đã scale và log)
# Đầu ra: Metrics RMSE, MAE, MAPE (đã inverse log) và Trực quan hóa.

# Nạp thư viện
library(tidyverse)
library(forecast)
library(torch)
library(ggplot2)

# Cấu hình giao diện và Seed
theme_set(theme_minimal(base_size = 14))
set.seed(42)
torch_manual_seed(42)

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN
# ---------------------------------------------------------
base_dir <- "C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality"
in_fs_dir <- paste0(base_dir, "/data/processed/modeling_fs/")
fig_dir   <- paste0(base_dir, "/outputs_R/figures/")

if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

TARGET_LOG <- "target_pm25_h24_log"

# ---------------------------------------------------------
# HÀM ĐÁNH GIÁ (INVERSE LOG SANG µg/m³)
# ---------------------------------------------------------
regression_metrics <- function(y_true, y_pred) {
  # Hoàn tác Log (expm1) để tính sai số trên thang gốc
  y_true_inv <- expm1(y_true)
  y_pred_inv <- expm1(y_pred)
  
  rmse <- sqrt(mean((y_true_inv - y_pred_inv)^2))
  mae  <- mean(abs(y_true_inv - y_pred_inv))
  eps  <- 1e-6
  mask <- abs(y_true_inv) > eps
  mape <- mean(abs((y_true_inv[mask] - y_pred_inv[mask]) / y_true_inv[mask])) * 100
  
  return(c(RMSE = rmse, MAE = mae, MAPE = mape))
}

# ---------------------------------------------------------
# 1. ĐỌC DỮ LIỆU
# ---------------------------------------------------------
cat(" Đang nạp dữ liệu...\n")
train_df <- read_csv(paste0(in_fs_dir, "train_dl.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
val_df   <- read_csv(paste0(in_fs_dir, "val_dl.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
test_df  <- read_csv(paste0(in_fs_dir, "test_dl.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))

feature_cols <- setdiff(colnames(train_df), c("datetime_local", TARGET_LOG))

X_train <- train_df[, feature_cols]
y_train <- train_df[[TARGET_LOG]]

X_val <- val_df[, feature_cols]
y_val <- val_df[[TARGET_LOG]]

X_test <- test_df[, feature_cols]
y_test <- test_df[[TARGET_LOG]]

test_dt <- train_df$datetime_local # Để phục vụ vẽ biểu đồ

cat(sprintf("Kích thước - Train: %d | Val: %d | Test: %d\n", nrow(train_df), nrow(val_df), nrow(test_df)))

# =====================================================================
# PHẦN 1: MÔ HÌNH THỐNG KÊ (ARIMA, SARIMA, ARIMAX, SARIMAX)
# =====================================================================
# Chọn 5 biến ngoại sinh (Exogenous variables) đầu tiên không phải họ pm25
exog_cols <- head(feature_cols[!grepl("pm25", feature_cols)], 5)

train_exog <- as.matrix(X_train[, exog_cols])
val_exog   <- as.matrix(X_val[, exog_cols])
test_exog  <- as.matrix(X_test[, exog_cols])

# Chuyển Target thành Time-Series object (Chu kỳ 24h = 1 ngày)
ts_train_y <- ts(y_train, frequency = 24)

stats_preds_train <- list()
stats_preds_val   <- list()
stats_preds_test  <- list()

# Hàm huấn luyện Thống kê
fit_and_predict_stats <- function(order, seasonal = c(0,0,0), use_exog = FALSE, name = "Model") {
  cat(sprintf(" Training %s...\n", name))
  
  xreg_train <- if (use_exog) train_exog else NULL
  xreg_val   <- if (use_exog) val_exog else NULL
  xreg_test  <- if (use_exog) test_exog else NULL
  
  # Fit mô hình trên Train
  model_fit <- Arima(ts_train_y, order = order, seasonal = list(order = seasonal, period = 24), xreg = xreg_train)
  stats_preds_train[[name]] <<- as.numeric(fitted(model_fit))
  
  # Apply mô hình lên Val & Test (1-step ahead dự báo)
  model_val <- Arima(y_val, model = model_fit, xreg = xreg_val)
  stats_preds_val[[name]] <<- as.numeric(fitted(model_val))
  
  model_test <- Arima(y_test, model = model_fit, xreg = xreg_test)
  stats_preds_test[[name]] <<- as.numeric(fitted(model_test))
  
  cat(sprintf("✅ %s fit xong!\n", name))
}

# FIT MODELS (Mô phỏng lại cấu hình từ Python)
fit_and_predict_stats(order = c(1,1,1), seasonal = c(0,0,0), use_exog = FALSE, name = "ARIMA")
fit_and_predict_stats(order = c(1,0,1), seasonal = c(1,0,0), use_exog = FALSE, name = "SARIMA")
fit_and_predict_stats(order = c(1,1,1), seasonal = c(0,0,0), use_exog = TRUE,  name = "ARIMAX")
fit_and_predict_stats(order = c(1,0,1), seasonal = c(1,0,0), use_exog = TRUE,  name = "SARIMAX")

# =====================================================================
# PHẦN 2: MÔ HÌNH HỌC SÂU (DEEP LEARNING: LSTM, GRU BẰNG TORCH)
# =====================================================================
cat("\n Đang chuẩn bị dữ liệu Tensor cho PyTorch...\n")

# Định nghĩa Device
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Device:", as.character(device), "\n")

# Chuyển đổi DataFrame sang Tensor có shape (Batch, Sequence=1, Features)
make_tensor <- function(X, y) {
  X_t <- torch_tensor(as.matrix(X), dtype = torch_float32())$unsqueeze(2)
  y_t <- torch_tensor(as.numeric(y), dtype = torch_float32())$unsqueeze(2)
  return(list(X = X_t, y = y_t))
}

train_t <- make_tensor(X_train, y_train)
val_t   <- make_tensor(X_val, y_val)
test_t  <- make_tensor(X_test, y_test)

# Dataloaders
batch_size <- 64
train_ds <- tensor_dataset(train_t$X, train_t$y)
val_ds   <- tensor_dataset(val_t$X, val_t$y)

train_loader <- dataloader(train_ds, batch_size = batch_size, shuffle = FALSE)
val_loader   <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)

num_features <- ncol(X_train)

# Định nghĩa Kiến trúc Mạng RNN (LSTM / GRU)
SimpleRNNModel <- nn_module(
  "SimpleRNNModel",
  initialize = function(rnn_type = "LSTM", input_size, hidden_size = 64, num_layers = 1) {
    self$rnn_type <- rnn_type
    if (rnn_type == "LSTM") {
      self$rnn <- nn_lstm(input_size, hidden_size, num_layers, batch_first = TRUE)
    } else {
      self$rnn <- nn_gru(input_size, hidden_size, num_layers, batch_first = TRUE)
    }
    self$fc <- nn_linear(hidden_size, 1)
  },
  forward = function(x) {
    out <- self$rnn(x)
    out <- out[[1]] # Lấy output Tensor, bỏ qua Hidden States
    # Chỉ lấy time-step cuối cùng
    out <- out[ , dim(out)[2], ] 
    out <- self$fc(out)
    return(out)
  }
)

# Hàm Huấn luyện (Training Loop)
train_dl_model <- function(model, epochs = 50, lr = 1e-3) {
  model <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = lr)
  criterion <- nn_mse_loss()
  
  for (epoch in 1:epochs) {
    model$train()
    train_loss <- 0
    
    coro::loop(for (b in train_loader) {
      X_b <- b[[1]]$to(device = device)
      y_b <- b[[2]]$to(device = device)
      
      optimizer$zero_grad()
      preds <- model(X_b)
      loss <- criterion(preds, y_b)
      loss$backward()
      optimizer$step()
      
      train_loss <- train_loss + loss$item() * X_b$size(1)
    })
    
    train_loss <- train_loss / length(train_ds)
    
    # Validation Loop
    model$eval()
    val_loss <- 0
    with_no_grad({
      coro::loop(for (b in val_loader) {
        X_b <- b[[1]]$to(device = device)
        y_b <- b[[2]]$to(device = device)
        preds <- model(X_b)
        loss <- criterion(preds, y_b)
        val_loss <- val_loss + loss$item() * X_b$size(1)
      })
    })
    val_loss <- val_loss / length(val_ds)
    
    if (epoch %% 10 == 0) {
      cat(sprintf("Epoch %02d | Train Loss: %.4f | Val Loss: %.4f\n", epoch, train_loss, val_loss))
    }
  }
  return(model)
}

# Hàm Dự báo
predict_dl <- function(model, X_tensor) {
  model$eval()
  with_no_grad({
    preds <- model(X_tensor$to(device = device))$cpu()$numpy()
  })
  return(as.numeric(preds))
}

# --- Chạy LSTM ---
cat("\n--- Training LSTM ---\n")
lstm_model <- SimpleRNNModel(rnn_type = "LSTM", input_size = num_features)
lstm_model <- train_dl_model(lstm_model, epochs = 50, lr = 0.001)

lstm_preds_train <- predict_dl(lstm_model, train_t$X)
lstm_preds_val   <- predict_dl(lstm_model, val_t$X)
lstm_preds_test  <- predict_dl(lstm_model, test_t$X)

# --- Chạy GRU ---
cat("\n--- Training GRU ---\n")
gru_model <- SimpleRNNModel(rnn_type = "GRU", input_size = num_features)
gru_model <- train_dl_model(gru_model, epochs = 50, lr = 0.001)

gru_preds_train <- predict_dl(gru_model, train_t$X)
gru_preds_val   <- predict_dl(gru_model, val_t$X)
gru_preds_test  <- predict_dl(gru_model, test_t$X)

# =====================================================================
# PHẦN 3: TỔNG HỢP VÀ ĐÁNH GIÁ
# =====================================================================
results <- data.frame()

# Đưa Thống kê vào bảng
for (m in names(stats_preds_train)) {
  r_tr <- regression_metrics(y_train, stats_preds_train[[m]])
  r_va <- regression_metrics(y_val, stats_preds_val[[m]])
  r_te <- regression_metrics(y_test, stats_preds_test[[m]])
  
  results <- rbind(results, 
                   data.frame(model = m, split = "train", RMSE = r_tr[1], MAE = r_tr[2], MAPE = r_tr[3]),
                   data.frame(model = m, split = "val", RMSE = r_va[1], MAE = r_va[2], MAPE = r_va[3]),
                   data.frame(model = m, split = "test", RMSE = r_te[1], MAE = r_te[2], MAPE = r_te[3]))
}

# Đưa DL vào bảng
dl_preds <- list(LSTM = list(train = lstm_preds_train, val = lstm_preds_val, test = lstm_preds_test),
                 GRU  = list(train = gru_preds_train, val = gru_preds_val, test = gru_preds_test))

for (m in names(dl_preds)) {
  r_tr <- regression_metrics(y_train, dl_preds[[m]]$train)
  r_va <- regression_metrics(y_val, dl_preds[[m]]$val)
  r_te <- regression_metrics(y_test, dl_preds[[m]]$test)
  
  results <- rbind(results, 
                   data.frame(model = m, split = "train", RMSE = r_tr[1], MAE = r_tr[2], MAPE = r_tr[3]),
                   data.frame(model = m, split = "val", RMSE = r_va[1], MAE = r_va[2], MAPE = r_va[3]),
                   data.frame(model = m, split = "test", RMSE = r_te[1], MAE = r_te[2], MAPE = r_te[3]))
}

# Hiện Test Metrics
cat("\n --- LEADERBOARD TRÊN TẬP TEST ---\n")
test_metrics <- results %>% filter(split == "test") %>% mutate(across(where(is.numeric), ~round(., 4)))
print(test_metrics)

# =====================================================================
# PHẦN 4: TRỰC QUAN HÓA (VISUALIZATION)
# =====================================================================
cat("\n Đang vẽ và lưu biểu đồ...\n")

# 4.1 Biểu đồ Cột Metrics
plot_df <- test_metrics %>% pivot_longer(cols = c(RMSE, MAE, MAPE), names_to = "metric", values_to = "value")
p_bar <- ggplot(plot_df, aes(x = metric, y = value, fill = model)) +
  geom_col(position = "dodge", color = "black", alpha = 0.8) +
  labs(title = "So sánh metric trên tập Test (PM2.5 t+24h)", x = "Metric", y = "Giá trị")

ggsave(paste0(fig_dir, "stats_dl_models_test_metrics_bar.png"), plot = p_bar, width = 10, height = 5)
print(p_bar)

# 4.2 Time-Series (200 giờ đầu)
subset_size <- min(200, length(y_test))
ts_df <- data.frame(
  Hour = 1:subset_size,
  Actual  = expm1(y_test[1:subset_size]),
  LSTM    = expm1(lstm_preds_test[1:subset_size]),
  GRU     = expm1(gru_preds_test[1:subset_size]),
  ARIMA   = expm1(stats_preds_test[["ARIMA"]][1:subset_size]),
  SARIMA  = expm1(stats_preds_test[["SARIMA"]][1:subset_size]),
  ARIMAX  = expm1(stats_preds_test[["ARIMAX"]][1:subset_size]),
  SARIMAX = expm1(stats_preds_test[["SARIMAX"]][1:subset_size])
) %>%
  pivot_longer(cols = -Hour, names_to = "Model", values_to = "PM2.5")

p_ts <- ggplot(ts_df, aes(x = Hour, y = PM2.5, color = Model, size = Model, linetype = Model)) +
  geom_line(alpha = 0.7) +
  scale_color_manual(values = c("Actual" = "black", "LSTM" = "blue", "GRU" = "cyan", 
                                "ARIMA" = "red", "SARIMA" = "orange", "ARIMAX" = "green", "SARIMAX" = "purple")) +
  scale_size_manual(values = c("Actual" = 1.5, "LSTM" = 0.8, "GRU" = 0.8, 
                               "ARIMA" = 0.8, "SARIMA" = 0.8, "ARIMAX" = 0.8, "SARIMAX" = 0.8)) +
  scale_linetype_manual(values = c("Actual" = "solid", "LSTM" = "solid", "GRU" = "solid", 
                                   "ARIMA" = "dashed", "SARIMA" = "dashed", "ARIMAX" = "dotted", "SARIMAX" = "dotted")) +
  labs(title = "Actual vs Predicted (200 giờ đầu của tập Test)", x = "Giờ", y = "PM2.5 (µg/m³)")

ggsave(paste0(fig_dir, "stats_dl_models_test_timeseries.png"), plot = p_ts, width = 14, height = 6)
print(p_ts)

cat(" Hoàn tất toàn bộ quy trình! Các biểu đồ đã được lưu.\n")