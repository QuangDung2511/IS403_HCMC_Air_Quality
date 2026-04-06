# --- LOAD THƯ VIỆN ---
library(tidyverse)
library(lubridate)
library(openair)   
library(gridExtra) 
library(forecast)  

# --- BƯỚC 1: LOAD DỮ LIỆU ---
# Lưu ý: Thay đổi đường dẫn tuyệt đối cho đúng với máy mỗi người
df <- read.csv("C:/Users/bao03/qbao_R/IS403_HCMC_Air_Quality/data/processed/hcmc_merged_cleaned.csv")
# Chuyển cột thời gian về định dạng chuẩn R
df$datetime_local <- as.POSIXct(df$datetime_local, format="%Y-%m-%d %H:%M:%S")

# Danh sách các biến cần phân tích theo yêu cầu
all_features <- c("pm25", "pm1", "temperature_2m", "relative_humidity_2m", 
                  "wind_speed_10m", "wind_direction_10m", "boundary_layer_height", 
                  "surface_pressure", "precipitation")

# --- BƯỚC 2: UNIVARIATE CHO TẤT CẢ CÁC BIẾN (Looping) ---
print("--- Đang thực hiện Univariate EDA ---")
plot_list <- list()

for (col in all_features) {
  # Biểu đồ phân phối (Histogram)
p1 <- ggplot(df, aes(x = .data[[col]])) +
  geom_histogram(fill = "#008080", color = "white", bins = 30) + # Mã Hex của màu Teal
  labs(title = paste("Distribution of", col)) + theme_minimal()
  
  # Biểu đồ ngoại lai (Boxplot)
  p2 <- ggplot(df, aes(y = .data[[col]])) +
    geom_boxplot(fill = "coral") +
    labs(title = paste("Outliers of", col)) + theme_minimal()
  
  plot_list[[length(plot_list) + 1]] <- p1
  plot_list[[length(plot_list) + 1]] <- p2
}

# Hiển thị lưới biểu đồ (Mỗi trang hiện 3 biến - 6 hình)
do.call(grid.arrange, c(plot_list[1:6], ncol = 2))

# --- BƯỚC 3: BIVARIATE VỚI REGRESSION LINES ---
print("--- Đang thực hiện Bivariate với Regression Lines ---")
p_wind <- ggplot(df, aes(x = wind_speed_10m, y = pm25)) +
  geom_point(alpha = 0.2, color = "gray") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "PM2.5 vs Wind Speed", x = "Wind Speed (m/s)", y = "PM2.5") + theme_minimal()

p_humid <- ggplot(df, aes(x = relative_humidity_2m, y = pm25)) +
  geom_point(alpha = 0.2, color = "gray") +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "PM2.5 vs Humidity", x = "Humidity (%)", y = "PM2.5") + theme_minimal()

grid.arrange(p_wind, p_humid, ncol = 2)

# --- BƯỚC 4: BINNING WIND DIRECTION & BOXPLOT ---
print("--- Phân tích Hướng gió bằng Boxplot ---")
# R có hàm cut cực mạnh để chia nhóm (Binning)
df$wind_dir_label <- cut(df$wind_direction_10m, 
                         breaks = seq(0, 360, by = 45),
                         labels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW"),
                         include.lowest = TRUE)

ggplot(df, aes(x = wind_dir_label, y = pm25, fill = wind_dir_label)) +
  geom_boxplot() +
  scale_fill_viridis_d() +
  labs(title = "PM2.5 Distribution by Wind Direction", x = "Direction", y = "PM2.5") +
  theme_minimal() + theme(legend.position = "none")

# --- BƯỚC 5: MULTIVARIATE - POLLUTION ROSE ---
print("--- Đang vẽ Pollution Rose (Sử dụng openair) ---")
#  openair yêu cầu cột tên 'ws' (wind speed) và 'wd' (wind direction)
df_rose <- df %>% rename(ws = wind_speed_10m, wd = wind_direction_10m, date = datetime_local)

pollutionRose(df_rose, pollutant = "pm25", 
              main = "HCMC Pollution Rose: PM2.5 & Wind Dynamics")

# --- BƯỚC 6: AUTOCORRELATION DECAY (48 Giờ) ---
ts_pm25 <- ts(na.omit(df$pm25), frequency = 24)
Acf(ts_pm25, lag.max = 48, main = "PM2.5 Autocorrelation Decay (48h)")
abline(v = 24, col = "red", lty = 2) # Kẻ vạch 24h để làm Insight