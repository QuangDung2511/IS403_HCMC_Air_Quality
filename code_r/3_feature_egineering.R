# =====================================================================
# BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG (Feature Engineering)
# =====================================================================
# Ánh xạ đầy đủ với Python notebook 3_feature_engineering.ipynb:
#   - extract_time_features()      : hour, day_of_week, month, is_weekend,
#                                    is_rush_hour, is_dry_season,
#                                    hour_sin/cos, month_sin/cos
#   - create_lags_and_rolling()    : pm25_lag_{1,2,3,6,12,24,48,72}
#                                    pm25_roll_{6,12,24,48,72}h_mean/std
#   - add_meteorological_physics() : is_raining, hours_since_last_rain,
#                                    wind_u, wind_v, wind_condition,
#                                    pm25_diff_1h, pm25_diff_24h,
#                                    pressure_trend_3h
#   - Target : target_pm25_h24 = lead(pm25, 24); dropna
# =====================================================================

# Nạp thư viện
library(tidyverse)
library(lubridate)

# =====================================================================
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN (động, không hard-code)
# =====================================================================
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

# Đường dẫn đọc dữ liệu (từ bước chia tập Train/Val/Test trước đó)
path_train <- file.path(PROJECT_ROOT, "data/processed_R/train_split.csv")
path_val   <- file.path(PROJECT_ROOT, "data/processed_R/val_split.csv")
path_test  <- file.path(PROJECT_ROOT, "data/processed_R/test_split.csv")

# Thư mục xuất dữ liệu dùng cho Modeling (thay thế modeling_fs)
out_dir <- file.path(PROJECT_ROOT, "data/processed_R/modeling_fs/")
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

# =====================================================================
# 2. HÀM TÍNH TOÁN KHOẢNG THỜI GIAN TẠNH MƯA
# =====================================================================
# Đếm số giờ liên tiếp KHÔNG có mưa kể từ lần mưa gần nhất.
# Tương đương với logic Python:
#   rain_mask = (precipitation > 0)
#   hours_since_last_rain = groupby(rain_mask.cumsum()).cumcount()
#   loc[is_raining == 1] = 0
calc_hours_since_rain <- function(precip) {
  is_raining <- (!is.na(precip)) & (precip > 0)
  # cumsum của nhóm mưa để tạo nhóm "dry spell"
  group_id <- cumsum(is_raining)
  # Đếm số hàng trong mỗi nhóm (0-indexed = số giờ kể từ lần mưa đầu nhóm)
  res <- ave(seq_along(group_id), group_id, FUN = function(x) seq_along(x) - 1L)
  # Các hàng mưa thực (is_raining = TRUE) → đặt lại về 0
  res[is_raining] <- 0L
  return(as.integer(res))
}

# =====================================================================
# 3. HÀM PHÂN LOẠI TỐC ĐỘ GIÓ (wind_condition)
# =====================================================================
# Khớp với Python: Calm / Light / Moderate+
categorize_wind <- function(speed) {
  case_when(
    speed < 0.5 ~ "Calm",
    speed < 2.0 ~ "Light",
    TRUE        ~ "Moderate+"
  )
}

# =====================================================================
# 4. HÀM TRÍCH XUẤT ĐẶC TRƯNG TỔNG HỢP (FEATURE ENGINEERING)
# =====================================================================
create_features <- function(df) {
  df <- df %>%
    mutate(datetime_local = ymd_hms(datetime_local)) %>%
    arrange(datetime_local)

  # ------------------------------------------------------------------
  # 4A. ĐẶC TRƯNG THỜI GIAN (Time Features)
  # ------------------------------------------------------------------
  # day_of_week: Python dùng dayofweek (0=Thứ 2, 6=Chủ Nhật)
  # → R: wday(week_start=7) trả về 1=CN...7=Th7; cần chuyển về 0-based Mon-first
  df <- df %>%
    mutate(
      hour        = hour(datetime_local),
      # 0 = Thứ Hai, 6 = Chủ Nhật (khớp với Python dayofweek)
      day_of_week = (wday(datetime_local, week_start = 7) + 5L) %% 7L,
      month       = month(datetime_local),

      # Cuối tuần: day_of_week >= 5 (Thứ 7 và Chủ Nhật)
      is_weekend  = as.integer(day_of_week >= 5),

      # Giờ cao điểm tại TP.HCM: 7-9h và 17-19h (khớp Python [7,8,9,17,18,19])
      is_rush_hour = as.integer(hour %in% c(7, 8, 9, 17, 18, 19)),

      # Mùa khô tại TP.HCM (Tháng 11 đến Tháng 4 năm sau)
      is_dry_season = as.integer(month %in% c(11, 12, 1, 2, 3, 4)),

      # Mã hóa chu kỳ cho Giờ (0-23) — khớp Python sin(2π*hour/24)
      hour_sin = sin(2 * pi * hour / 24),
      hour_cos = cos(2 * pi * hour / 24),

      # Mã hóa chu kỳ cho Tháng — khớp Python sin(2π*(month-1)/12)
      month_sin = sin(2 * pi * (month - 1) / 12),
      month_cos = cos(2 * pi * (month - 1) / 12)
    )

  # ------------------------------------------------------------------
  # 4B. ĐẶC TRƯNG KHÍ TƯỢNG (Meteorological Physics)
  # ------------------------------------------------------------------
  df <- df %>%
    mutate(
      # Biến nhị phân: có mưa không?
      is_raining = as.integer(!is.na(precipitation) & precipitation > 0),

      # Số giờ tạnh mưa liên tiếp
      hours_since_last_rain = calc_hours_since_rain(precipitation),

      # Phân rã vector gió thành 2 trục U (Đông-Tây) và V (Bắc-Nam)
      # Python: wind_u = speed * cos(rad), wind_v = speed * sin(rad)
      wind_u = wind_speed_10m * cos(wind_direction_10m * pi / 180),
      wind_v = wind_speed_10m * sin(wind_direction_10m * pi / 180),

      # Phân loại tốc độ gió
      wind_condition = categorize_wind(wind_speed_10m),

      # Đà thay đổi PM2.5 và áp suất (Momentum / Trends)
      pm25_diff_1h       = pm25 - lag(pm25, 1),
      pm25_diff_24h      = pm25 - lag(pm25, 24),
      pressure_trend_3h  = surface_pressure - lag(surface_pressure, 3)
    )

  # ------------------------------------------------------------------
  # 4C. ĐẶC TRƯNG TRỄ (Lag Features)
  # Python: lags = [1, 2, 3, 6, 12, 24, 48, 72]
  # ------------------------------------------------------------------
  for (lag_n in c(1, 2, 3, 6, 12, 24, 48, 72)) {
    df[[paste0("pm25_lag_", lag_n)]] <- lag(df$pm25, lag_n)
  }

  # ------------------------------------------------------------------
  # 4D. ĐẶC TRƯNG CUỘN (Rolling Window Features)
  # Python: windows = [6, 12, 24, 48, 72]
  # ------------------------------------------------------------------
  for (w in c(6, 12, 24, 48, 72)) {
    df[[paste0("pm25_roll_", w, "h_mean")]] <- slider::slide_dbl(
      df$pm25, mean, .before = w - 1, .complete = TRUE
    )
    df[[paste0("pm25_roll_", w, "h_std")]] <- slider::slide_dbl(
      df$pm25, sd, .before = w - 1, .complete = TRUE
    )
  }

  # ------------------------------------------------------------------
  # 4E. BIẾN MỤC TIÊU (Target Variable: Dự báo t+24h)
  # ------------------------------------------------------------------
  df <- df %>%
    mutate(target_pm25_h24 = lead(pm25, 24)) %>%
    # Loại bỏ mọi hàng có NA (do lag, rolling, hoặc target lead)
    drop_na(target_pm25_h24)

  return(df)
}

# =====================================================================
# 5. THỰC THI VÀ LƯU KẾT QUẢ
# =====================================================================
cat("Đang đọc và xử lý các tập dữ liệu...\n")

# Kiểm tra thư viện slider
if (!requireNamespace("slider", quietly = TRUE)) {
  install.packages("slider")
}
library(slider)

train_df <- read_csv(path_train, show_col_types = FALSE)
val_df   <- read_csv(path_val,   show_col_types = FALSE)
test_df  <- read_csv(path_test,  show_col_types = FALSE)

train_fe <- create_features(train_df)
val_fe   <- create_features(val_df)
test_fe  <- create_features(test_df)

cat("Kích thước sau khi tạo đặc trưng:\n")
cat("Train: ", dim(train_fe)[1], "dòng, ", dim(train_fe)[2], "cột\n")
cat("Val:   ", dim(val_fe)[1],   "dòng, ", dim(val_fe)[2],   "cột\n")
cat("Test:  ", dim(test_fe)[1],  "dòng, ", dim(test_fe)[2],  "cột\n")

# Lưu các file đã Feature Engineering vào thư mục modeling_fs
# (Đây là đầu vào của bước 5 - Feature Selection)
write_csv(train_fe, file.path(out_dir, "train_fe.csv"))
write_csv(val_fe,   file.path(out_dir, "val_fe.csv"))
write_csv(test_fe,  file.path(out_dir, "test_fe.csv"))

cat("Đã hoàn tất! Các file FE đã được lưu tại:\n", out_dir, "\n")