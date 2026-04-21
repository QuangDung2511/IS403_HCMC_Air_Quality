# =====================================================================
# BƯỚC 13: MÔ HÌNH KẾT HỢP (HYBRID SARIMAX + GRU)
# =====================================================================

library(tidyverse)
library(torch)
library(coro)
library(Metrics)
library(ggplot2)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

in_fs_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
pred_dir  <- file.path(PROJECT_ROOT, "outputs_R/predictions/")
fig_dir   <- file.path(PROJECT_ROOT, "outputs_R/figures/")

TARGET_LOG <- "target_pm25_h24_log"
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Device:", as.character(device), "\n")

regression_metrics <- function(y_true, y_pred) {
  rmse_val <- sqrt(mean((y_true - y_pred)^2))
  mae_val  <- mean(abs(y_true - y_pred))
  eps <- 1e-6
  mask <- y_true > eps
  mape_val <- mean(abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
  return(data.frame(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val))
}

# ---------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------
load_data <- function(split) {
  df <- read_csv(file.path(in_fs_dir, paste0(split, "_dl.csv")), show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
  feat_cols <- setdiff(colnames(df), c("datetime_local", TARGET_LOG))
  X <- as.matrix(df[, feat_cols])
  y_real <- expm1(df[[TARGET_LOG]])
  return(list(X=X, y=y_real, feat_cols=feat_cols, df=df))
}

d_train <- load_data("train")
d_val   <- load_data("val")
d_test  <- load_data("test")
num_features <- length(d_train$feat_cols)

# Load SARIMAX predictions
f_stats <- file.path(pred_dir, "tuned_stats_preds_real_units.rds")
if (!file.exists(f_stats)) stop("Cần chạy 9_tuning_stats.R trước!")
stats_preds <- readRDS(f_stats)

sar_train <- stats_preds$SARIMAX_train
sar_val   <- stats_preds$SARIMAX_val
sar_test  <- stats_preds$SARIMAX_test

# Compute residuals (ground truth - sarimax)
resid_train <- d_train$y - sar_train
resid_val   <- d_val$y - sar_val

cat(sprintf("Residual stats (Train) – Mean: %.3f, Std: %.3f\n", mean(resid_train), sd(resid_train)))
cat(sprintf("Residual stats (Val)   – Mean: %.3f, Std: %.3f\n", mean(resid_val), sd(resid_val)))

# ---------------------------------------------------------
# MAKE TENSORS
# ---------------------------------------------------------
make_loader <- function(X, y_resid, shuffle=FALSE) {
  X_t <- torch_tensor(X, dtype=torch_float32())$unsqueeze(2)
  y_t <- torch_tensor(as.numeric(y_resid), dtype=torch_float32())$unsqueeze(2)
  ds <- tensor_dataset(X_t, y_t)
  return(dataloader(ds, batch_size=64, shuffle=shuffle))
}

train_loader <- make_loader(d_train$X, resid_train, shuffle=TRUE)
val_loader   <- make_loader(d_val$X, resid_val,   shuffle=FALSE)
test_loader  <- make_loader(d_test$X, rep(0, nrow(d_test$X)), shuffle=FALSE) # dummy y

# ---------------------------------------------------------
# DEFINE GRU RESIDUAL MODEL
# ---------------------------------------------------------
ResidualGRU <- nn_module(
  "ResidualGRU",
  initialize = function(input_size, hidden_size=64, num_layers=1, bidirectional=TRUE, dropout=0.1) {
    self$gru <- nn_gru(input_size, hidden_size, num_layers, batch_first=TRUE,
                       dropout=if(num_layers>1) dropout else 0, bidirectional=bidirectional)
    
    fc_in <- if(bidirectional) hidden_size * 2 else hidden_size
    self$fc <- nn_sequential(
      nn_linear(fc_in, fc_in %/% 2),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(fc_in %/% 2, 1)
    )
  },
  forward = function(x) {
    out <- self$gru(x)
    # Get last time step
    out <- out[[1]][, dim(out[[1]])[2], ]
    return(self$fc(out))
  }
)

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
train_gru <- function(model, epochs=120, lr=5e-4, patience=12) {
  model <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr=lr, weight_decay=1e-5)
  scheduler <- lr_reduce_on_plateau(optimizer, mode='min', patience=5, factor=0.5)
  criterion <- nn_mse_loss()
  
  best_val_loss <- Inf
  p_counter <- 0
  train_history <- c()
  val_history   <- c()
  best_weights  <- NULL
  
  for (epoch in 1:epochs) {
    model$train()
    t_loss <- 0
    total_samples <- 0
    coro::loop(for (b in train_loader) {
      optimizer$zero_grad()
      preds <- model(b[[1]]$to(device = device))
      loss  <- criterion(preds, b[[2]]$to(device = device))
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, 1.0)
      optimizer$step()
      t_loss <- t_loss + (loss$item() * dim(b[[1]])[1])
      total_samples <- total_samples + dim(b[[1]])[1]
    })
    t_loss <- t_loss / total_samples
    
    model$eval()
    v_loss <- 0
    total_val <- 0
    with_no_grad({
      coro::loop(for (b in val_loader) {
        preds <- model(b[[1]]$to(device = device))
        loss  <- criterion(preds, b[[2]]$to(device = device))
        v_loss <- v_loss + (loss$item() * dim(b[[1]])[1])
        total_val <- total_val + dim(b[[1]])[1]
      })
    })
    v_loss <- v_loss / total_val
    
    scheduler$step(v_loss)
    train_history <- c(train_history, t_loss)
    val_history   <- c(val_history, v_loss)
    
    if (v_loss < best_val_loss - 1e-4) {
      best_val_loss <- v_loss
      best_weights  <- model$state_dict()
      p_counter <- 0
    } else {
      p_counter <- p_counter + 1
    }
    
    if (epoch %% 10 == 0) {
      cat(sprintf("Epoch %3d/%d | Train MSE: %.5f | Val MSE: %.5f\n", epoch, epochs, t_loss, v_loss))
    }
    
    if (p_counter >= patience) {
      cat(sprintf("\nEarly stopping at epoch %d.\n", epoch))
      break
    }
  }
  
  model$load_state_dict(best_weights)
  return(list(model=model, hist=list(train=train_history, val=val_history)))
}

cat("Bắt đầu huấn luyện GRU dự báo phần dư...\n")
model <- ResidualGRU(input_size=num_features, hidden_size=64, num_layers=1, bidirectional=TRUE, dropout=0.1)
res <- train_gru(model, epochs=120, lr=5e-4, patience=12)
model <- res$model

# ---------------------------------------------------------
# PREDICTIONS & EXPORT
# ---------------------------------------------------------
predict_gru <- function(X) {
  model$eval()
  with_no_grad({
    X_t <- torch_tensor(X, dtype=torch_float32())$unsqueeze(2)
    return(as.numeric(model(X_t$to(device = device))$cpu()))
  })
}

gru_resid_train <- predict_gru(d_train$X)
gru_resid_val   <- predict_gru(d_val$X)
gru_resid_test  <- predict_gru(d_test$X)

hybrid_train <- sar_train + gru_resid_train
hybrid_val   <- sar_val + gru_resid_val
hybrid_test  <- sar_test + gru_resid_test

results <- bind_rows(
  data.frame(Split="Train", Model="SARIMAX", regression_metrics(d_train$y, sar_train)),
  data.frame(Split="Train", Model="Hybrid", regression_metrics(d_train$y, hybrid_train)),
  data.frame(Split="Val", Model="SARIMAX", regression_metrics(d_val$y, sar_val)),
  data.frame(Split="Val", Model="Hybrid", regression_metrics(d_val$y, hybrid_val)),
  data.frame(Split="Test", Model="SARIMAX", regression_metrics(d_test$y, sar_test)),
  data.frame(Split="Test", Model="Hybrid", regression_metrics(d_test$y, hybrid_test))
)

cat("\n=== Final Test Metrics ===\n")
print(results %>% filter(Split == "Test"))

hybrid_export <- list(
  hybrid_sarimax_gru = list(
    train = hybrid_train,
    val   = hybrid_val,
    test  = hybrid_test,
    note  = "Real-unit µg/m³"
  )
)

saveRDS(hybrid_export, file.path(pred_dir, "hybrid_sarimax_gru_preds.rds"))
cat("\nSaved hybrid predictions to hybrid_sarimax_gru_preds.rds\n")
