# --- BƯỚC 0: TẢI CÔNG CỤ ---
library(tidyverse)
library(lubridate)
library(zoo) # Để tính Rolling Window

# --- BƯỚC 1: LOAD DỮ LIỆU ---
# Bảo nhớ Set Working Directory đến thư mục chứa file trước khi chạy nhé
df <- read.csv("hcmc_merged_cleaned.csv")
df$datetime_local <- as.POSIXct(df$datetime_local, format="%Y-%m-%d %H:%M:%S")

# --- BƯỚC 2: DROP BIẾN (Theo yêu cầu của bảng) ---
df_clean <- df %>%
  # 1. DROP pm1, um003 (Tránh Data Leakage)
  # 2. DROP temperature, relativehumidity từ OpenAQ (Nhiễu)
  select(-any_of(c("pm1", "um003", "temperature", "relativehumidity"))) %>%
  # Loại bỏ dòng trắng ở biến mục tiêu
  filter(!is.na(pm25))

# --- BƯỚC 3: TRANSFORMATIONS (Log Transformation) ---
# Dùng log1p (np.log1p trong Python) cho pm25 và boundary_layer_height
df_clean$pm25_log <- log1p(df_clean$pm25)
df_clean$blh_log <- log1p(df_clean$boundary_layer_height)

# --- BƯỚC 4: FEATURE CREATION (Áp suất & Mưa) ---
df_clean <- df_clean %>%
  arrange(datetime_local) %>%
  mutate(
    # 1. Áp suất: Tính Delta P (P_t - P_{t-3})
    delta_pressure = surface_pressure - lag(surface_pressure, 3),
    
    # 2. Mưa: Tạo flag is_raining (0/1)
    is_raining = ifelse(precipitation > 0, 1, 0),
    
    # 3. Mưa: Tính số giờ kể từ lần mưa cuối (hours_since_last_rain)
    # Tạo biến phụ để đếm
    rain_group = cumsum(is_raining)
  ) %>%
  group_by(rain_group) %>%
  mutate(hours_since_last_rain = row_number() - 1) %>%
  ungroup() %>%
  select(-rain_group)

# --- BƯỚC 5: XỬ LÝ GIÓ (Phân rã thành U-wind và V-wind) ---
df_clean <- df_clean %>%
  mutate(
    u_wind = wind_speed_10m * cos(wind_direction_10m * pi / 180),
    v_wind = wind_speed_10m * sin(wind_direction_10m * pi / 180)
  ) %>%
  select(-wind_direction_10m) # DROP hướng gió cũ theo yêu cầu

# --- BƯỚC 6: CYCLICAL ENCODING (Giờ, Ngày trong tuần, Tháng) ---
df_clean <- df_clean %>%
  mutate(
    hour = hour(datetime_local),
    day_of_week = wday(datetime_local),
    month = month(datetime_local),
    # Encoding Sin/Cos cho Hour và Month
    hour_sin = sin(2 * pi * hour / 24),
    hour_cos = cos(2 * pi * hour / 24),
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12)
  )

# --- BƯỚC 7: LAG FEATURES & ROLLING WINDOW ---
df_final <- df_clean %>%
  mutate(
    # Lag Features: t-1, t-2, t-24
    pm25_lag1 = lag(pm25, 1),
    pm25_lag2 = lag(pm25, 2),
    pm25_lag24 = lag(pm25, 24),
    # Rolling Window: Mean 6h, 12h
    pm25_roll_6h = rollmean(pm25, k = 6, fill = NA, align = "right"),
    pm25_roll_12h = rollmean(pm25, k = 12, fill = NA, align = "right")
  ) %>%
  drop_na() # Xóa dòng NA phát sinh do tạo Lag

# --- BƯỚC 8: SCALING (StandardScaler) ---
# Áp dụng cho các biến thời tiết vĩ mô theo yêu cầu
weather_vars <- c("temperature_2m", "relative_humidity_2m", "surface_pressure", "blh_log")
df_final[weather_vars] <- lapply(df_final[weather_vars], scale)

# --- BƯỚC 9: XUẤT FILE ---
write.csv(df_final, "hcmc_final_preprocessed.csv", row.names = FALSE)

print("--- CHÚC MỪNG! CODE ĐÃ CHẠY KHỚP VỚI YÊU CẦU ---")