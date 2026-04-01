# --- LOAD THƯ VIỆN ---
library(tidyverse)
library(lubridate)
library(zoo) # Hỗ trợ tính Rolling Window

# --- BƯỚC 1: LOAD DỮ LIỆU ---
df <- read.csv("hcmc_merged_cleaned.csv")
df$datetime_local <- as.POSIXct(df$datetime_local, format="%Y-%m-%d %H:%M:%S")

# --- BƯỚC 2: DROP BIẾN (Theo yêu cầu "DROP hoàn toàn") ---
df_clean <- df %>%
  select(-c(pm1, um003)) %>% # Tránh Data Leakage & Đa cộng tuyến
  # Drop các biến từ OpenAQ nếu có, ưu tiên OpenMeteo theo bảng
  select(-any_of(c("temperature", "relativehumidity"))) 

# --- BƯỚC 3: TRANSFORMATIONS (Log & Math) ---

# 1. Log Transformation (Cho pm25 và boundary_layer_height)
# Dùng log1p để xử lý nồng độ thấp và ổn định phương sai
df_clean$pm25_log <- log1p(df_clean$pm25)
df_clean$blh_log <- log1p(df_clean$boundary_layer_height)

# 2. Xử lý Áp suất: Tính Delta P (P_t - P_{t-3})
df_clean <- df_clean %>%
  mutate(delta_pressure = surface_pressure - lag(surface_pressure, 3))

# --- BƯỚC 4: FEATURE CREATION (Mưa & Gió) ---

# 1. Mưa (Zero-inflated): Tạo Flag is_raining
df_clean <- df_clean %>%
  mutate(is_raining = ifelse(precipitation > 0, 1, 0))

# 2. Gió: Phân rã thành U-wind (Đông-Tây) và V-wind (Bắc-Nam)
# Công thức: U = speed * cos(direction), V = speed * sin(direction)
df_clean <- df_clean %>%
  mutate(
    u_wind = wind_speed_10m * cos(wind_direction_10m * pi / 180),
    v_wind = wind_speed_10m * sin(wind_direction_10m * pi / 180)
  ) %>%
  select(-wind_direction_10m) # DROP theo yêu cầu

# --- BƯỚC 5: CYCLICAL ENCODING (Cho Hour & Month) ---
# Giúp mô hình hiểu được 23h đêm rất gần với 0h sáng
df_clean <- df_clean %>%
  mutate(
    hour = hour(datetime_local),
    month = month(datetime_local),
    # Encoding Sin/Cos
    hour_sin = sin(2 * pi * hour / 24),
    hour_cos = cos(2 * pi * hour / 24),
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12)
  )

# --- BƯỚC 6: LAG FEATURES & ROLLING WINDOW ---
df_final <- df_clean %>%
  mutate(
    # Lag Features (t-1, t-2, t-24)
    pm25_lag1 = lag(pm25, 1),
    pm25_lag2 = lag(pm25, 2),
    pm25_lag24 = lag(pm25, 24),
    # Rolling Window (Mean 6h, 12h)
    pm25_roll_6h = rollmean(pm25, k = 6, fill = NA, align = "right"),
    pm25_roll_12h = rollmean(pm25, k = 12, fill = NA, align = "right")
  ) %>%
  drop_na() # Loại bỏ các dòng NA tạo ra do Lag

# Xuất file chuẩn bị cho Model
# write.csv(df_final, "hcmc_preprocessed_ready.csv", row.names = FALSE)