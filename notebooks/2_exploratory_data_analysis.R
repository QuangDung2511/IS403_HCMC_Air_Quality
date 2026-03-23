# 0. Cài đặt các thư viện cần thiết (nếu chưa có)
# install.packages(c("tidyverse", "lubridate", "corrplot", "tseries", "forecast", "gridExtra"))

library(tidyverse)
library(lubridate)
library(corrplot)
library(tseries)
library(forecast)
library(gridExtra)

# 1. Load dữ liệu
# Lưu ý: RStudio tự hiểu Working Directory nếu bạn mở theo Project
df <- read.csv("C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality/data/processed/hcmc_merged_cleaned.csv")
# 2. Xử lý thời gian
df$datetime_local <- as.POSIXct(df$datetime_local, format="%Y-%m-%d %H:%M:%S")

# Kiểm tra dữ liệu
print("✅ Đã load dữ liệu thành công!")
glimpse(df)

# 2.1. Check missing values
print("--- Kiểm tra giá trị thiếu ---")
colSums(is.na(df))

# 2.2. Check Distribution & Outliers
p1 <- ggplot(df, aes(x = pm25)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "teal", alpha = 0.6) +
  geom_density(color = "red", size = 1) +
  labs(title = "Phân phối nồng độ PM2.5", x = "PM2.5", y = "Mật độ") +
  theme_minimal()

p2 <- ggplot(df, aes(y = pm25)) +
  geom_boxplot(fill = "coral", alpha = 0.7) +
  labs(title = "Kiểm tra Outliers của PM2.5", y = "PM2.5") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

# 2.3. Kiểm tra tính dừng (ADF Test)
print("--- Kiểm định tính dừng (ADF Test) ---")
adf_test <- adf.test(na.omit(df$pm25))
print(adf_test)
# Giải thích: p-value <= 0.05 là dữ liệu có tính dừng.

#----------------------------------------------------------------------

# 2.1. Check missing values
print("--- Kiểm tra giá trị thiếu ---")
colSums(is.na(df))

# 2.2. Check Distribution & Outliers
p1 <- ggplot(df, aes(x = pm25)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "teal", alpha = 0.6) +
  geom_density(color = "red", size = 1) +
  labs(title = "Phân phối nồng độ PM2.5", x = "PM2.5", y = "Mật độ") +
  theme_minimal()

p2 <- ggplot(df, aes(y = pm25)) +
  geom_boxplot(fill = "coral", alpha = 0.7) +
  labs(title = "Kiểm tra Outliers của PM2.5", y = "PM2.5") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

# 2.3. Kiểm tra tính dừng (ADF Test)
print("--- Kiểm định tính dừng (ADF Test) ---")
adf_test <- adf.test(na.omit(df$pm25))
print(adf_test)
# Giải thích: p-value <= 0.05 là dữ liệu có tính dừng.

#----------------------------------------------------------------------

# 3.1. Numerical Correlation (Heatmap)
# Chỉ chọn các cột số
df_numeric <- df %>% select_if(is.numeric)
cor_matrix <- cor(df_numeric, use = "complete.obs")

corrplot(cor_matrix, method = "color", addCoef.col = "black", 
         tl.col = "black", number.cex = 0.7,
         title = "\nMa trận tương quan giữa các biến số", mar = c(0,0,1,0))

# 3.2. PM2.5 theo Giờ trong ngày
df$hour <- hour(df$datetime_local)

df_hourly <- df %>%
  group_by(hour) %>%
  summarise(
    mean_pm25 = mean(pm25, na.rm = TRUE),
    sd_pm25 = sd(pm25, na.rm = TRUE),
    n = n()
  ) %>%
  mutate(se = sd_pm25 / sqrt(n)) # Sai số chuẩn cho khoảng tin cậy

ggplot(df_hourly, aes(x = hour, y = mean_pm25)) +
  geom_line(color = "red", size = 1) +
  geom_point(color = "red") +
  geom_ribbon(aes(ymin = mean_pm25 - 1.96*se, ymax = mean_pm25 + 1.96*se), 
              fill = "red", alpha = 0.2) +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Xu hướng PM2.5 theo giờ (Trung bình & CI 95%)", x = "Giờ", y = "PM2.5 (µg/m³)") +
  theme_minimal()

# 3.3. Tự tương quan (ACF/PACF)
# Tạo đối tượng Time Series (tần suất 24h)
ts_pm25 <- ts(df$pm25, frequency = 24)
par(mfrow = c(1, 2))
Acf(ts_pm25, main = "ACF PM2.5 (Lag 48h)", lag.max = 48)
Pacf(ts_pm25, main = "PACF PM2.5 (Lag 48h)", lag.max = 48)
par(mfrow = c(1, 1))

#----------------------------------------------------------------------

# 4.1. Heatmap Giờ vs Thứ
df$day_name <- wday(df$datetime_local, label = TRUE, abbr = FALSE, week_start = 1)

df_heatmap <- df %>%
  group_by(day_name, hour) %>%
  summarise(avg_pm25 = mean(pm25, na.rm = TRUE))

ggplot(df_heatmap, aes(x = hour, y = day_name, fill = avg_pm25)) +
  geom_tile() +
  scale_fill_gradient(low = "yellow", high = "red") +
  labs(title = "Cường độ ô nhiễm: Giờ vs Thứ trong tuần", x = "Giờ trong ngày", y = "Thứ", fill = "PM2.5") +
  theme_minimal()

# 4.2. Phân rã chuỗi thời gian (Decomposition)
decomp <- stl(ts_pm25, s.window = "periodic")
plot(decomp, main = "Phân rã chuỗi thời gian PM2.5")

# 4.3. Check Class Balance (AQI)
df <- df %>%
  mutate(aqi_label = case_when(
    pm25 <= 15 ~ "Tốt",
    pm25 <= 35 ~ "Trung bình",
    pm25 <= 55 ~ "Kém",
    TRUE ~ "Xấu/Rất xấu"
  ))

df_aqi_count <- df %>%
  count(aqi_label) %>%
  mutate(prop = n / sum(n) * 100)

ggplot(df_aqi_count, aes(x = "", y = prop, fill = aqi_label)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  labs(title = "Tỷ lệ các mức độ chất lượng không khí") +
  theme_void() +
  scale_fill_manual(values = c("Tốt"="lightgreen", "Trung bình"="gold", "Kém"="orange", "Xấu/Rất xấu"="red"))