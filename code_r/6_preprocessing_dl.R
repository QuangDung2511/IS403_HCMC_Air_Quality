# ---------------------------------------------------------------------
# BƯỚC 6: TIỀN XỬ LÝ CHO MÔ HÌNH (Preprocessing cho DL / Thống kê)
# ---------------------------------------------------------------------

library(tidyverse)
library(jsonlite)

# ---------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN (động, không hard-code)
# ---------------------------------------------------------
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

data_dir   <- file.path(PROJECT_ROOT, "data/processed_R/")

# Đọc từ thư mục R của chính mình (processed_R/modeling_fs/)
in_fs_dir  <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")

# Thư mục đích (Lưu kết quả của R)
out_fs_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
if (!dir.exists(out_fs_dir)) {
  dir.create(out_fs_dir, recursive = TRUE, showWarnings = FALSE)
}

TARGET_ORIG <- "target_pm25_h24"
TARGET_LOG  <- "target_pm25_h24_log"

# ---------------------------------------------------------
# 1. ĐỌC DỮ LIỆU TỪ BƯỚC FEATURE SELECTION
# ---------------------------------------------------------
cat("Đang nạp dữ liệu...\n")
train_tree <- read_csv(file.path(in_fs_dir, "train_tree.csv"), show_col_types = FALSE)
val_tree   <- read_csv(file.path(in_fs_dir, "val_tree.csv"),   show_col_types = FALSE)
test_tree  <- read_csv(file.path(in_fs_dir, "test_tree.csv"),  show_col_types = FALSE)

feature_cols <- setdiff(colnames(train_tree), c("datetime_local", TARGET_ORIG))

is_pm_family_column <- function(name) {
  name == "pm25" || startsWith(name, "pm25_")
}
log_feature_cols <- feature_cols[sapply(feature_cols, is_pm_family_column)]

# ---------------------------------------------------------
# 2. BIẾN ĐỔI LOG (log1p)
# ---------------------------------------------------------
apply_log1p <- function(x) {
  x <- ifelse(x < 0, 0, x)
  log1p(x)
}

train_t <- train_tree %>% mutate(across(all_of(log_feature_cols), apply_log1p))
val_t   <- val_tree   %>% mutate(across(all_of(log_feature_cols), apply_log1p))
test_t  <- test_tree  %>% mutate(across(all_of(log_feature_cols), apply_log1p))

train_t[[TARGET_LOG]] <- apply_log1p(train_tree[[TARGET_ORIG]])
val_t[[TARGET_LOG]]   <- apply_log1p(val_tree[[TARGET_ORIG]])
test_t[[TARGET_LOG]]  <- apply_log1p(test_tree[[TARGET_ORIG]])

# ---------------------------------------------------------
# 3. CHUẨN HÓA (STANDARD SCALER)
# ---------------------------------------------------------
cat("⏳ Đang chuẩn hóa (Scaling) dữ liệu...\n")

X_train <- as.matrix(train_t[, feature_cols])
X_val   <- as.matrix(val_t[, feature_cols])
X_test  <- as.matrix(test_t[, feature_cols])

# Dùng scale() của Base R — tương đương sklearn StandardScaler
X_train_s    <- scale(X_train)
train_center <- attr(X_train_s, "scaled:center")
train_scale  <- attr(X_train_s, "scaled:scale")

X_val_s  <- scale(X_val,  center = train_center, scale = train_scale)
X_test_s <- scale(X_test, center = train_center, scale = train_scale)

# ---------------------------------------------------------
# 4. ĐÓNG GÓI VÀ LƯU XUẤT (BUILD & EXPORT)
# ---------------------------------------------------------
build_out_df <- function(dt, X_scaled, feat_names, target_log_series) {
  out <- as.data.frame(X_scaled)
  colnames(out) <- feat_names
  out <- out %>%
    mutate(
      datetime_local    = dt,
      !!TARGET_LOG      := target_log_series
    ) %>%
    select(datetime_local, all_of(feat_names), all_of(TARGET_LOG))
  return(out)
}

train_dl <- build_out_df(as.character(train_t$datetime_local), X_train_s, feature_cols, train_t[[TARGET_LOG]])
val_dl   <- build_out_df(as.character(val_t$datetime_local),   X_val_s,   feature_cols, val_t[[TARGET_LOG]])
test_dl  <- build_out_df(as.character(test_t$datetime_local),  X_test_s,  feature_cols, test_t[[TARGET_LOG]])

write_csv(train_dl, file.path(out_fs_dir, "train_dl.csv"))
write_csv(val_dl,   file.path(out_fs_dir, "val_dl.csv"))
write_csv(test_dl,  file.path(out_fs_dir, "test_dl.csv"))

cat(sprintf("✅ Đã ghi CSV tại thư mục: %s\n", out_fs_dir))

# ---------------------------------------------------------
# 5. LƯU THÔNG SỐ TIỀN XỬ LÝ (META-DATA)
# ---------------------------------------------------------
scaler_info <- list(
  center              = train_center,
  scale               = train_scale,
  feature_columns     = feature_cols,
  log1p_feature_columns = log_feature_cols
)
saveRDS(scaler_info, file.path(data_dir, "preprocessing_dl_scaler.rds"))

meta <- list(
  generated_at_utc    = format(Sys.time(), tz = "UTC", usetz = TRUE),
  source_splits       = "modeling_fs/train_tree.csv, val_tree.csv, test_tree.csv",
  target_original     = TARGET_ORIG,
  target_transformed  = TARGET_LOG,
  log1p_feature_columns = log_feature_cols,
  scaler              = "Base R scale() (center & scale vectors)",
  inverse_target      = "expm1(target_pm25_h24_log)"
)
write_json(meta, file.path(data_dir, "preprocessing_dl_meta.json"), pretty = TRUE, auto_unbox = TRUE)

# ---------------------------------------------------------
# 6. KIỂM TRA ĐẢO NGƯỢC (SANITY CHECK)
# ---------------------------------------------------------
cat("--- Kiểm tra Inverse (Sanity Check) ---\n")
orig_target    <- head(train_tree[[TARGET_ORIG]], 3)
logged_target  <- head(train_dl[[TARGET_LOG]], 3)
restored_target <- expm1(logged_target)

cat("Gốc: ",            orig_target,    "\n")
cat("expm1(log): ",     restored_target, "\n")
cat("Sai số trung bình: ", max(abs(orig_target - restored_target)), "\n")