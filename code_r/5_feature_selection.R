# =====================================================================
# BƯỚC 5: LỰA CHỌN ĐẶC TRƯNG (Feature Selection) - Dự báo PM2.5
# =====================================================================
# Mục tiêu:
# - Xếp hạng mức độ quan trọng của đặc trưng bằng Random Forest và XGBoost.
# - Chọn Top-K đặc trưng, có tùy chọn ép giữ các biến trễ (lags) từ PACF.

# Nạp thư viện
library(tidyverse)
library(randomForest)
library(xgboost)
library(jsonlite)
library(ggplot2)

# Thiết lập giao diện biểu đồ
theme_set(theme_minimal(base_size = 14))

# ---------------------------------------------------------
# CẤU HÌNH BAN ĐẦU (Configuration)
# ---------------------------------------------------------
RANDOM_STATE <- 42
TOP_K        <- 25
FORCE_INCLUDE_PACF_LAGS <- TRUE

# Các cột loại bỏ theo phân tích EDA (khớp Python)
EDA_DROP_COLS <- c("pm1", "um003", "temperature", "relativehumidity", "wind_direction_10m")

# Các cột buộc phải giữ lại theo PACF
FORCE_LAG_FEATURES <- c("pm25_lag_1", "pm25_lag_2", "pm25_lag_24")

# ---------------------------------------------------------
# ĐỊNH NGHĨA ĐƯỜNG DẪN (động, không hard-code)
# ---------------------------------------------------------
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

# Đọc từ kết quả FE của bước 3 (processed_R/modeling_fs/train_fe.csv ...)
in_fe_dir  <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
out_fs_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
fig_dir    <- file.path(PROJECT_ROOT, "outputs_R/figures/")

if (!dir.exists(fig_dir))    dir.create(fig_dir,    recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(out_fs_dir)) dir.create(out_fs_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------
# 1. ĐỌC DỮ LIỆU & TIỀN XỬ LÝ NHANH
# ---------------------------------------------------------
cat("⏳ Đang nạp dữ liệu...\n")

train_df <- read_csv(file.path(in_fe_dir, "train_fe.csv"), show_col_types = FALSE)
val_df   <- read_csv(file.path(in_fe_dir, "val_fe.csv"),   show_col_types = FALSE)
test_df  <- read_csv(file.path(in_fe_dir, "test_fe.csv"),  show_col_types = FALSE)

# Loại bỏ các biến đa cộng tuyến/nhiễu
drop_cols_func <- function(df, cols) {
  df %>% select(-any_of(cols))
}

train_df <- drop_cols_func(train_df, EDA_DROP_COLS)
val_df   <- drop_cols_func(val_df,   EDA_DROP_COLS)
test_df  <- drop_cols_func(test_df,  EDA_DROP_COLS)

# Mã hóa wind_condition thành số (Label Encoding):
# Chỉ dùng levels từ tập Train để tránh data leakage (khớp Python LabelEncoder).
wind_levels <- sort(unique(train_df$wind_condition))

encode_wind <- function(df, levels) {
  if ("wind_condition" %in% colnames(df)) {
    df <- df %>%
      mutate(wind_condition_encoded = as.numeric(factor(wind_condition, levels = levels)) - 1L) %>%
      select(-wind_condition)
  }
  return(df)
}

train_df <- encode_wind(train_df, wind_levels)
val_df   <- encode_wind(val_df,   wind_levels)
test_df  <- encode_wind(test_df,  wind_levels)

# Xác định danh sách biến đặc trưng (Features)
# Bỏ qua cột thời gian và biến mục tiêu
feature_cols <- setdiff(colnames(train_df), c("datetime_local", "target_pm25_h24"))
cat("Số đặc trưng ứng viên:", length(feature_cols), "\n")

# Chuẩn bị ma trận huấn luyện — bỏ NA ở target
train_clean <- train_df %>% drop_na(target_pm25_h24)
X_train <- as.data.frame(train_clean[, feature_cols])
y_train <- train_clean$target_pm25_h24

# ---------------------------------------------------------
# 2. XẾP HẠNG BẰNG RANDOM FOREST & XGBOOST
# ---------------------------------------------------------
cat("Đang huấn luyện Random Forest (ntree=200)...\n")
set.seed(RANDOM_STATE)
# ntree=200, max_depth không được RF base R hỗ trợ trực tiếp; dùng options tương đương
rf_model <- randomForest(x = X_train, y = y_train, ntree = 200, importance = TRUE)

# Trích xuất tầm quan trọng (%IncMSE)
imp_rf <- importance(rf_model, type = 1)[, 1]

cat("Đang huấn luyện XGBoost (nrounds=200)...\n")
# Chuẩn bị dữ liệu cho chuẩn DMatrix của XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_params <- list(
  max_depth         = 6,
  eta               = 0.08,    # learning_rate khớp Python
  objective         = "reg:squarederror",
  tree_method       = "hist",
  seed              = RANDOM_STATE
)

xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 200)

# Lấy tầm quan trọng (Gain)
xgb_imp_table <- xgb.importance(model = xgb_model)
imp_xgb_named <- setNames(xgb_imp_table$Gain, xgb_imp_table$Feature)
imp_xgb <- sapply(feature_cols, function(x) ifelse(is.na(imp_xgb_named[x]), 0, imp_xgb_named[x]))

# ---------------------------------------------------------
# 3. CHUẨN HÓA VÀ XẾP HẠNG (Min-Max Scaling & Ranking)
# ---------------------------------------------------------
minmax_norm <- function(v) {
  lo <- min(v)
  hi <- max(v)
  if (hi - lo < 1e-12) return(rep(1, length(v)))
  return((v - lo) / (hi - lo))
}

norm_rf  <- minmax_norm(imp_rf)
norm_xgb <- minmax_norm(imp_xgb)

# Tính hạng — Điểm càng cao -> thứ hạng càng nhỏ (1 là tốt nhất)
rank_rf  <- rank(-imp_rf,  ties.method = "average")
rank_xgb <- rank(-imp_xgb, ties.method = "average")
mean_rank <- (rank_rf + rank_xgb) / 2.0

# Tạo bảng tổng hợp
ranking_df <- data.frame(
  feature             = feature_cols,
  importance_rf       = imp_rf,
  importance_xgb      = imp_xgb,
  norm_importance_rf  = norm_rf,
  norm_importance_xgb = norm_xgb,
  rank_rf             = rank_rf,
  rank_xgb            = rank_xgb,
  mean_rank           = mean_rank
) %>%
  arrange(mean_rank)

# ---------------------------------------------------------
# 4. CHỌN TOP-K VÀ ÉP GIỮ PACF LAGS
# ---------------------------------------------------------
ordered_features  <- ranking_df$feature
selected_features <- head(ordered_features, TOP_K)

forced_added <- c()
if (FORCE_INCLUDE_PACF_LAGS) {
  for (f in FORCE_LAG_FEATURES) {
    if (!(f %in% selected_features) && (f %in% feature_cols)) {
      selected_features <- c(selected_features, f)
      forced_added      <- c(forced_added, f)
    }
  }
}

cat(sprintf("Top_K = %d, số đặc trưng chính thức được chọn: %d\n", TOP_K, length(selected_features)))
if (length(forced_added) > 0) {
  cat("Đã ép thêm (PACF):", paste(forced_added, collapse = ", "), "\n")
}

# ---------------------------------------------------------
# 5. TRỰC QUAN HÓA (VISUALIZATION)
# ---------------------------------------------------------
top_n <- min(20, nrow(ranking_df))

plot_df <- ranking_df %>% head(top_n) %>% arrange(desc(mean_rank)) %>%
  mutate(feature = factor(feature, levels = feature))

p_rank <- ggplot(plot_df, aes(x = mean_rank, y = feature)) +
  geom_col(fill = "seagreen") +
  labs(title   = sprintf("Trung bình thứ hạng RF + XGB - Top %d", top_n),
       x = "Mean rank (Nhỏ hơn = Quan trọng hơn)", y = "")

ggsave(file.path(fig_dir, "feature_importance_mean_rank.png"), plot = p_rank, width = 10, height = 8)
print(p_rank)

# ---------------------------------------------------------
# 6. XUẤT FILE BÁO CÁO VÀ DỮ LIỆU
# ---------------------------------------------------------
# Lưu bảng Ranking
data_dir <- file.path(PROJECT_ROOT, "data/processed_R/")
ranking_path <- file.path(data_dir, "feature_importance_ranking.csv")
write_csv(ranking_df, ranking_path)

# Tạo và lưu Meta-data JSON
meta <- list(
  generated_at        = format(Sys.time(), tz = "UTC", usetz = TRUE),
  top_k               = TOP_K,
  eda_dropped_columns = EDA_DROP_COLS,
  force_include_pacf_lags = FORCE_INCLUDE_PACF_LAGS,
  forced_lag_features = forced_added,
  selected_features   = selected_features
)

json_path <- file.path(data_dir, "selected_features.json")
write_json(meta, json_path, pretty = TRUE, auto_unbox = TRUE)

# Lưu CSV rút gọn cho Modeling (train_tree / val_tree / test_tree)
out_cols <- c("datetime_local", "target_pm25_h24", selected_features)

write_csv(train_df %>% select(all_of(out_cols)), file.path(out_fs_dir, "train_tree.csv"))
write_csv(val_df   %>% select(all_of(out_cols)), file.path(out_fs_dir, "val_tree.csv"))
write_csv(test_df  %>% select(all_of(out_cols)), file.path(out_fs_dir, "test_tree.csv"))

cat(" Đã ghi toàn bộ CSV tại:", out_fs_dir, "\n")