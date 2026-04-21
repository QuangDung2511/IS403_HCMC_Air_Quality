# =====================================================================
# BƯỚC 12: SO SÁNH TOÀN BỘ MÔ HÌNH
# =====================================================================

library(tidyverse)
library(Metrics)
library(ggplot2)
library(jsonlite)

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

in_fs_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
pred_dir  <- file.path(PROJECT_ROOT, "outputs_R/predictions/")
fig_dir   <- file.path(PROJECT_ROOT, "outputs_R/figures/")

if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

TARGET_LOG <- "target_pm25_h24_log"

cat("Loading Test Ground Truth...\n")
test_df <- read_csv(file.path(in_fs_dir, "test_dl.csv"), show_col_types = FALSE) %>% drop_na(all_of(TARGET_LOG))
y_test_real <- expm1(test_df[[TARGET_LOG]])

regression_metrics <- function(y_true, y_pred) {
  rmse_val <- sqrt(mean((y_true - y_pred)^2))
  mae_val  <- mean(abs(y_true - y_pred))
  eps <- 1e-6
  mask <- y_true > eps
  mape_val <- mean(abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
  return(data.frame(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val))
}

results <- data.frame()
preds_dict <- list()

# 1. Thống Kê (Real units)
cat("Loading Stats predictions...\n")
f_stats <- file.path(pred_dir, "tuned_stats_preds_real_units.rds")
if (file.exists(f_stats)) {
  stats_preds <- readRDS(f_stats)
  for (m in c("ARIMA", "ARIMAX", "SARIMA", "SARIMAX")) {
    k <- paste0(m, "_test")
    if (!is.null(stats_preds[[k]])) {
      pred_real <- stats_preds[[k]]
      preds_dict[[m]] <- pred_real
      mets <- regression_metrics(y_test_real, pred_real)
      results <- bind_rows(results, data.frame(Model = m, Group = "Stats", mets))
    }
  }
}

# 2. Deep Learning (Log space -> convert)
cat("Loading DL predictions...\n")
f_dl <- file.path(pred_dir, "tuned_dl_preds.rds")
if (file.exists(f_dl)) {
  dl_preds <- readRDS(f_dl)
  pred_real <- expm1(as.numeric(dl_preds$test))
  
  model_name <- "GRU_Tuned"
  if (!is.null(dl_preds$params)) {
    model_name <- paste0(dl_preds$params$rnn_type, "_Tuned")
    if (dl_preds$params$bidirectional) model_name <- paste0("Bi", model_name)
  }
  preds_dict[[model_name]] <- pred_real
  mets <- regression_metrics(y_test_real, pred_real)
  results <- bind_rows(results, data.frame(Model = model_name, Group = "DL", mets))
}

# 3. Tree (Real units)
cat("Loading Tree predictions...\n")
f_tree <- file.path(pred_dir, "tuned_tree_preds.rds")
if (file.exists(f_tree)) {
  tree_preds <- readRDS(f_tree)
  for (m in names(tree_preds)) {
    pred_real <- as.numeric(tree_preds[[m]])
    preds_dict[[m]] <- pred_real
    mets <- regression_metrics(y_test_real, pred_real)
    results <- bind_rows(results, data.frame(Model = m, Group = "Tree", mets))
  }
}

# 4. Hybrid
cat("Loading Hybrid predictions...\n")
f_hybrid <- file.path(pred_dir, "hybrid_sarimax_gru_preds.rds")
if (file.exists(f_hybrid)) {
  hybrid_preds <- readRDS(f_hybrid)
  if (!is.null(hybrid_preds$hybrid_sarimax_gru$test)) {
    pred_real <- hybrid_preds$hybrid_sarimax_gru$test
    m <- "Hybrid"
    preds_dict[[m]] <- pred_real
    mets <- regression_metrics(y_test_real, pred_real)
    results <- bind_rows(results, data.frame(Model = m, Group = "Hybrid", mets))
  }
}

cat("\n--- ALL MODELS COMPARISON ---\n")
final_table <- results %>% arrange(RMSE)
print(final_table)
write_csv(final_table, file.path(pred_dir, "all_models_comparison_metrics.csv"))

# Visualizations
plot_df <- final_table %>% pivot_longer(cols = c(RMSE, MAE, MAPE), names_to = "Metric", values_to = "Value")
p1 <- ggplot(plot_df, aes(x = reorder(Model, Value), y = Value, fill = Group)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Metric, scales = "free_y") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Model Comparison", x = "Model", y = "Value")
ggsave(file.path(fig_dir, "all_models_comparison_metrics.png"), p1, width = 12, height = 5)

cat("\nHoàn tất.\n")
