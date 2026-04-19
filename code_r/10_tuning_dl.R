# =====================================================================
# BƯỚC 10: TINH CHỈNH MÔ HÌNH DEEP LEARNING (LSTM & GRU)
# =====================================================================
# Mục tiêu: Thực hiện Grid Search tìm kiến trúc RNN tốt nhất cho PM2.5.
# Các tham số tuning: Loại RNN, Hidden Size, Number of Layers, Bidirectional, Dropout.

# Nạp thư viện
library(tidyverse)
library(torch)
library(coro)
library(Metrics)

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN (động, không hard-code)
# ---------------------------------------------------------
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

# Đọc từ thư mục R của chính mình (processed_R/modeling_fs/)
in_fs_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
pred_dir  <- file.path(PROJECT_ROOT, "outputs_R/predictions/")

if (!dir.exists(pred_dir)) dir.create(pred_dir, recursive = TRUE)

TARGET_LOG <- "target_pm25_h24_log"
device     <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Phần cứng đang sử dụng:", as.character(device), "\n")

# ---------------------------------------------------------
# HÀM TRỢ GIÚP (Metrics & Loader)
# ---------------------------------------------------------
regression_metrics <- function(y_true, y_pred) {
  y_true_inv <- expm1(as.numeric(y_true))
  y_pred_inv <- expm1(as.numeric(y_pred))

  rmse_val <- sqrt(mean((y_true_inv - y_pred_inv)^2))
  mae_val  <- mean(abs(y_true_inv - y_pred_inv))

  eps <- 1e-6
  mask <- abs(y_true_inv) > eps
  mape_val <- mean(abs((y_true_inv[mask] - y_pred_inv[mask]) / y_true_inv[mask])) * 100

  return(c(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val))
}

# Đọc dữ liệu
train_df <- read_csv(file.path(in_fs_dir, "train_dl.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
val_df   <- read_csv(file.path(in_fs_dir, "val_dl.csv"),   show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
test_df  <- read_csv(file.path(in_fs_dir, "test_dl.csv"),  show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))

feature_cols <- setdiff(colnames(train_df), c("datetime_local", TARGET_LOG))
num_features <- length(feature_cols)

# Tạo Tensors (Batch, Seq=1, Features)
make_tensor_dataset <- function(df) {
  X <- torch_tensor(as.matrix(df[, feature_cols]), dtype = torch_float32())$unsqueeze(2)
  y <- torch_tensor(as.numeric(df[[TARGET_LOG]]),  dtype = torch_float32())$unsqueeze(2)
  return(tensor_dataset(X, y))
}

train_ds <- make_tensor_dataset(train_df)
val_ds   <- make_tensor_dataset(val_df)
test_ds  <- make_tensor_dataset(test_df)

train_loader <- dataloader(train_ds, batch_size = 64, shuffle = FALSE)
val_loader   <- dataloader(val_ds,   batch_size = 64, shuffle = FALSE)

# =====================================================================
# 1. KIẾN TRÚC MẠNG RNN LINH HOẠT (DynamicRNN)
# =====================================================================
DynamicRNN <- nn_module(
  "DynamicRNN",
  initialize = function(rnn_type="LSTM", input_size, hidden_size, num_layers, dropout, bidirectional) {
    self$bidirectional <- bidirectional
    self$num_layers    <- num_layers

    if (rnn_type == "LSTM") {
      self$rnn <- nn_lstm(input_size, hidden_size, num_layers, batch_first = TRUE,
                          dropout = if(num_layers > 1) dropout else 0, bidirectional = bidirectional)
    } else {
      self$rnn <- nn_gru(input_size, hidden_size, num_layers, batch_first = TRUE,
                         dropout = if(num_layers > 1) dropout else 0, bidirectional = bidirectional)
    }

    fc_dim <- if(bidirectional) hidden_size * 2 else hidden_size
    self$fc <- nn_sequential(
      nn_linear(fc_dim, fc_dim %/% 2),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(fc_dim %/% 2, 1)
    )
  },
  forward = function(x) {
    out <- self$rnn(x)
    # out[[1]] là output của tất cả time-steps, lấy step cuối cùng
    out <- out[[1]][, dim(out[[1]])[2], ]
    return(self$fc(out))
  }
)

# =====================================================================
# 2. VÒNG LẶP HUẤN LUYỆN VỚI EARLY STOPPING
# =====================================================================
train_eval_model <- function(model, epochs=60, lr=1e-3, patience=7) {
  model     <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = 1e-5)
  scheduler <- lr_reduce_on_plateau(optimizer, mode = 'min', patience = 3, factor = 0.5)
  criterion <- nn_mse_loss()

  best_val_loss    <- Inf
  patience_counter <- 0

  for (epoch in 1:epochs) {
    model$train()
    coro::loop(for (b in train_loader) {
      optimizer$zero_grad()
      preds <- model(b[[1]]$to(device = device))
      loss  <- criterion(preds, b[[2]]$to(device = device))
      loss$backward()
      optimizer$step()
    })

    # Eval
    model$eval()
    val_loss <- 0
    with_no_grad({
      coro::loop(for (b in val_loader) {
        preds    <- model(b[[1]]$to(device = device))
        val_loss <- val_loss + criterion(preds, b[[2]]$to(device = device))$item()
      })
    })
    val_loss <- val_loss / length(val_loader)
    scheduler$step(val_loss)

    if (val_loss < best_val_loss) {
      best_val_loss    <- val_loss
      patience_counter <- 0
    } else {
      patience_counter <- patience_counter + 1
    }

    if (patience_counter >= patience) break
  }
  return(model)
}

# =====================================================================
# 3. ARCHITECTURE GRID SEARCH
# =====================================================================
param_grid <- expand.grid(
  rnn_type      = c("LSTM", "GRU"),
  hidden_size   = c(64, 128),
  num_layers    = c(1, 2),
  bidirectional = c(FALSE, TRUE),
  dropout       = c(0.1, 0.3),
  stringsAsFactors = FALSE
)

cat("Tổng số cấu hình cần thử nghiệm:", nrow(param_grid), "\n")

results_list      <- list()
best_rmse         <- Inf
best_params       <- NULL
best_model_weights <- NULL

for (i in 1:nrow(param_grid)) {
  p <- param_grid[i, ]
  cat(sprintf("[%d/%d] Đang thử: %s, Hid:%d, Lay:%d, Bi:%s, Drp:%.1f\n",
              i, nrow(param_grid), p$rnn_type, p$hidden_size, p$num_layers, p$bidirectional, p$dropout))

  # Khởi tạo và Train
  model_trial <- DynamicRNN(p$rnn_type, num_features, p$hidden_size, p$num_layers, p$dropout, p$bidirectional)
  model_trial <- train_eval_model(model_trial)

  # Dự báo trên tập Val để lấy RMSE thật (Inverse Log)
  model_trial$eval()
  with_no_grad({
    X_val_t  <- val_ds$.getbatch(1:length(val_ds))[[1]]
    val_preds <- as.numeric(model_trial(X_val_t$to(device = device))$cpu())
  })

  metrics <- regression_metrics(val_df[[TARGET_LOG]], val_preds)
  cat(sprintf("   => Val RMSE: %.4f\n", metrics["RMSE"]))

  if (metrics["RMSE"] < best_rmse) {
    best_rmse          <- metrics["RMSE"]
    best_params        <- p
    best_model_weights <- model_trial$state_dict()
  }
}

cat("\n=== GRID SEARCH HOÀN TẤT ===\n")
print(best_params)

# =====================================================================
# 4. CHẠY MODEL TỐT NHẤT TRÊN TẬP TEST & LƯU TRỮ
# =====================================================================
best_model <- DynamicRNN(best_params$rnn_type, num_features, best_params$hidden_size,
                         best_params$num_layers, best_params$dropout, best_params$bidirectional)
best_model$load_state_dict(best_model_weights)
best_model$to(device = device)

# Dự báo cuối cùng
get_preds <- function(ds) {
  best_model$eval()
  with_no_grad({
    X <- ds$.getbatch(1:length(ds))[[1]]
    return(as.numeric(best_model(X$to(device = device))$cpu()))
  })
}

dl_preds <- list(
  train  = get_preds(train_ds),
  val    = get_preds(val_ds),
  test   = get_preds(test_ds),
  params = best_params
)

# Đánh giá trên tập Test
cat("\n--- HIỆU SUẤT TRÊN TẬP TEST (BEST DL) ---\n")
print(regression_metrics(test_df[[TARGET_LOG]], dl_preds$test))

# Lưu kết quả
saveRDS(dl_preds, file.path(pred_dir, "tuned_dl_preds.rds"))
cat("\n✅ Đã lưu kết quả dự báo tại:", file.path(pred_dir, "tuned_dl_preds.rds\n"))