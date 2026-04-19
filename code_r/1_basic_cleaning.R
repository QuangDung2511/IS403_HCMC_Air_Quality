# =====================================================================
# BƯỚC 1: LÀM SẠCH DỮ LIỆU CƠ BẢN (Basic Cleaning)
# =====================================================================

# Nạp các thư viện cần thiết
library(tidyverse)
library(lubridate)
library(zoo)

# 1. Thiết lập đường dẫn hệ thống (động, không hard-code)
# Lấy thư mục gốc của project (thư mục cha của code_r/)
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

path_air     <- file.path(PROJECT_ROOT, "data/raw/hcm_air_data.csv")
path_weather <- file.path(PROJECT_ROOT, "data/raw/hcmc_weather_data.csv")

# Đường dẫn lưu trữ kết quả riêng cho luồng xử lý bằng R
path_output  <- file.path(PROJECT_ROOT, "data/processed_R/hcmc_merged_cleaned.csv")

# 2. Đọc dữ liệu thô
air_df     <- read_csv(path_air, show_col_types = FALSE)
weather_df <- read_csv(path_weather, show_col_types = FALSE)

# 3. Tiền xử lý thời gian (Datetime)
air_df <- air_df %>%
  mutate(
    datetime_local = ymd_hms(datetime_local),
    # Chuyển đổi về dạng naive (loại bỏ múi giờ) để đồng nhất với định dạng của dữ liệu thời tiết
    datetime_local_naive = force_tz(datetime_local, tzone = "UTC")
  )

weather_df <- weather_df %>%
  mutate(time = ymd_hms(time))

# 4. Kết hợp hai bộ dữ liệu (Merge) dựa trên mốc thời gian
df <- air_df %>%
  left_join(weather_df, by = c("datetime_local_naive" = "time")) %>%
  select(-datetime_local_naive) %>%
  arrange(datetime_local)

# 5. Loại bỏ các đặc trưng không phù hợp
# Loại bỏ pm10 và temperature_180m do tỷ lệ dữ liệu trống (NaN) vượt ngưỡng cho phép
df <- df %>%
  select(-any_of(c("pm10", "temperature_180m")))

# 6. Chuẩn hóa tần suất dữ liệu (Resample) và Nội suy (Interpolate)
# Tạo khung thời gian liên tục theo từng giờ (Hourly)
full_time_sequence <- seq(
  from = min(df$datetime_local, na.rm = TRUE),
  to   = max(df$datetime_local, na.rm = TRUE),
  by   = "hour"
)

df_interpolated <- df %>%
  distinct(datetime_local, .keep_all = TRUE) %>%
  complete(datetime_local = full_time_sequence) %>%
  # Sử dụng nội suy tuyến tính (Linear Interpolation) để xử lý các giá trị thiếu
  # rule = 2 giúp xử lý các giá trị NA ở biên (đầu và cuối chuỗi)
  mutate(across(where(is.numeric), ~ na.approx(.x, na.rm = FALSE, rule = 2)))

# 7. Lưu trữ kết quả vào thư mục processed_R
# Tự động tạo thư mục nếu chưa tồn tại trong hệ thống
if (!dir.exists(dirname(path_output))) {
  dir.create(dirname(path_output), recursive = TRUE, showWarnings = FALSE)
}

write_csv(df_interpolated, path_output)

cat("Hoàn tất Bước 1. Dữ liệu đã được lưu tại luồng R:", path_output, "\n")