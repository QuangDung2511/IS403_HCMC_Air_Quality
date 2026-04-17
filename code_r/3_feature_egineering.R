# =====================================================================
# BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG (Feature Engineering)
# =====================================================================

# Nạp thư viện
library(tidyverse)
library(lubridate)

# =====================================================================
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN
# =====================================================================
base_dir <- "C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality"

# Đường dẫn đọc dữ liệu (từ bước chia tập Train/Val/Test trước đó)
path_train <- paste0(base_dir, "/data/processed_R/train_split.csv")
path_val   <- paste0(base_dir, "/data/processed_R/val_split.csv")
path_test  <- paste0(base_dir, "/data/processed_R/test_split.csv")

# Thư mục xuất dữ liệu dùng cho Modeling
out_dir <- paste0(base_dir, "/data/processed_R/modeling_fs/")
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

# =====================================================================
# 2. HÀM TÍNH TOÁN KHOẢNG THỜI GIAN TẠNH MƯA
# =====================================================================
# Hàm này đếm số giờ trôi qua kể từ lần có lượng mưa (precipitation) > 0 gần nhất
calc_hours_since_rain <- function(precip) {
  res <- numeric(length(precip))
  counter <- 0
  for (i in seq_along(precip)) {
    if (!is.na(precip[i]) && precip[i] > 0) {
      counter <- 0 # Reset bộ đếm nếu có mưa
    } else {
      counter <- counter + 1 # Tăng số giờ tạnh mưa
    }
    res[i] <- counter
  }
  return(res)
}

# =====================================================================
# 3. HÀM TRÍCH XUẤT ĐẶC TRƯNG TỔNG HỢP (FEATURE ENGINEERING)
# =====================================================================
create_features <- function(df) {
  df <- df %>%
    # Đảm bảo cột thời gian đúng định dạng
    mutate(datetime_local = ymd_hms(datetime_local)) %>%
    arrange(datetime_local) %>%
    
    mutate(
      # --- ĐẶC TRƯNG THỜI GIAN (Time Features) ---
      hour = hour(datetime_local),
      day_of_week = wday(datetime_local, week_start = 1), # 1 = Thứ 2, 7 = Chủ Nhật
      month = month(datetime_local),
      
      # Mùa khô tại TP.HCM (Tháng 11 đến Tháng 4 năm sau)
      is_dry_season = ifelse(month %in% c(11, 12, 1, 2, 3, 4), 1, 0),
      
      # Mã hóa chu kỳ (Cyclical Encoding) cho Giờ và Tháng bằng hàm Lượng giác
      hour_sin = sin(2 * pi * hour / 24),
      hour_cos = cos(2 * pi * hour / 24),
      month_sin = sin(2 * pi * month / 12),
      month_cos = cos(2 * pi * month / 12),
      
      # --- ĐẶC TRƯNG THỜI TIẾT (Weather Features) ---
      # Phân rã vector gió thành 2 trục U (Đông-Tây) và V (Bắc-Nam)
      wind_u = wind_speed_10m * sin(wind_direction_10m * pi / 180),
      wind_v = wind_speed_10m * cos(wind_direction_10m * pi / 180),
      
      # Tính số giờ tạnh mưa liên tục
      hours_since_last_rain = calc_hours_since_rain(precipitation)
    ) %>%
    
    # --- TẠO BIẾN MỤC TIÊU (Target Variable: Dự báo t+24h) ---
    # Lấy giá trị PM2.5 của 24 giờ sau mang về hàng hiện tại
    mutate(
      target_pm25_h24 = lead(pm25, 24),
      # Log transform (log1p) để giảm độ lệch (skewness) của Target
      target_pm25_h24_log = log1p(target_pm25_h24)
    ) %>%
    
    # Do hàm lead(24) sẽ tạo ra 24 giá trị NA ở cuối bảng, ta cần loại bỏ chúng
    drop_na(target_pm25_h24_log)
  
  return(df)
}

# =====================================================================
# 4. THỰC THI VÀ LƯU KẾT QUẢ
# =====================================================================
cat("Đang đọc và xử lý các tập dữ liệu...\n")

train_df <- read_csv(path_train, show_col_types = FALSE)
val_df   <- read_csv(path_val, show_col_types = FALSE)
test_df  <- read_csv(path_test, show_col_types = FALSE)

train_fe <- create_features(train_df)
val_fe   <- create_features(val_df)
test_fe  <- create_features(test_df)

cat("Kích thước sau khi tạo đặc trưng:\n")
cat("Train: ", dim(train_fe)[1], "dòng, ", dim(train_fe)[2], "cột\n")
cat("Val:   ", dim(val_fe)[1], "dòng, ", dim(val_fe)[2], "cột\n")
cat("Test:  ", dim(test_fe)[1], "dòng, ", dim(test_fe)[2], "cột\n")

# Lưu các file đã Feature Engineering vào thư mục modeling_fs
write_csv(train_fe, paste0(out_dir, "train_dl.csv"))
write_csv(val_fe,   paste0(out_dir, "val_dl.csv"))
write_csv(test_fe,  paste0(out_dir, "test_dl.csv"))

cat("Đã hoàn tất! Các file sẵn sàng cho mô hình đã được lưu tại:\n", out_dir, "\n")