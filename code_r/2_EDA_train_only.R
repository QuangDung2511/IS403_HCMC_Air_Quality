# --------------------------------------------------------------------=
# BƯỚC 2: KHÁM PHÁ DỮ LIỆU (Exploratory Data Analysis - EDA)
# --------------------------------------------------------------------=

# 1. NẠP THƯ VIỆN
library(tidyverse)
library(lubridate)
library(ggplot2)
library(corrplot)
library(forecast)

# Cài đặt giao diện chuẩn (Theme) cho tất cả các biểu đồ
theme_set(theme_minimal(base_size = 14))

print("✅ Đã nạp thư viện thành công!")

# --------------------------------------------------------------------=
# 2. ĐỊNH NGHĨA ĐƯỜNG DẪN (Cấu trúc thư mục luồng R)
# --------------------------------------------------------------------=
# Lấy thư mục gốc của project (thư mục cha của code_r/)
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), winslash = "/")

# Đường dẫn tập Train đã được chia (Lấy từ folder processed_R)
data_path <- file.path(PROJECT_ROOT, "data/processed_R/train_split.csv")

# Thư mục lưu trữ hình ảnh (Lưu vào outputs_R)
fig_dir <- file.path(PROJECT_ROOT, "outputs_R/figures/")

# Đảm bảo thư mục lưu hình ảnh tồn tại
if (!dir.exists(fig_dir)) {
    dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
}

# Đọc dữ liệu
train_df <- read_csv(data_path, show_col_types = FALSE)

# Định dạng lại cột thời gian và sắp xếp
train_df <- train_df %>%
    mutate(datetime_local = ymd_hms(datetime_local)) %>%
    arrange(datetime_local)

cat("Kích thước tập Train: ", nrow(train_df), "dòng, ", ncol(train_df), "cột\n")

# --------------------------------------------------------------------=
# 3. PHÂN TÍCH BIẾN MỤC TIÊU (TARGET VARIABLE: PM2.5)
# --------------------------------------------------------------------=
print("⏳ Đang phân tích và vẽ biểu đồ phân phối PM2.5...")

# 3.1 Biểu đồ phân phối (Histogram & Density)
p_dist <- ggplot(train_df, aes(x = pm25)) +
    geom_histogram(aes(y = after_stat(density)), bins = 50, fill = "steelblue", color = "white", alpha = 0.8) +
    geom_density(color = "red", linewidth = 1.2) +
    labs(
        title = "Phân phối nồng độ PM2.5 (Tập Train)",
        subtitle = "Kiểm tra độ lệch (Skewness) của dữ liệu",
        x = "Nồng độ PM2.5 (µg/m³)",
        y = "Mật độ (Density)"
    )

# Lưu và hiển thị
ggsave(paste0(fig_dir, "pm25_distribution.png"), plot = p_dist, width = 8, height = 5)
print(p_dist)

# 3.2 Biểu đồ chuỗi thời gian (Time-Series Plot)
p_ts <- ggplot(train_df, aes(x = datetime_local, y = pm25)) +
    geom_line(color = "darkslategray", alpha = 0.8, linewidth = 0.5) +
    labs(
        title = "Biến động PM2.5 theo thời gian",
        subtitle = "Quan sát xu hướng tổng thể và các đỉnh nhiễu (Spikes)",
        x = "Thời gian",
        y = "Nồng độ PM2.5 (µg/m³)"
    )

ggsave(paste0(fig_dir, "pm25_timeseries.png"), plot = p_ts, width = 12, height = 5)
print(p_ts)

# --------------------------------------------------------------------=
# 4. PHÂN TÍCH TƯƠNG QUAN (CORRELATION HEATMAP)
# --------------------------------------------------------------------=
print("⏳ Đang tính toán ma trận tương quan Pearson...")

# Lọc các biến số học (Numeric) để tính tương quan
numeric_df <- train_df %>% select(where(is.numeric))

# Xử lý NA tạm thời (pairwise.complete.obs) để tránh lỗi khi tính toán
cor_matrix <- cor(numeric_df, use = "pairwise.complete.obs", method = "pearson")

# Cấu hình màu cho Heatmap (Đỏ: Tương quan âm, Xanh: Tương quan dương)
col_palette <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

# Lưu biểu đồ ra file
png(paste0(fig_dir, "correlation_heatmap.png"), width = 900, height = 900, res = 100)
corrplot(cor_matrix,
    method = "color", type = "upper",
    tl.col = "black", tl.srt = 45,
    addCoef.col = "black", number.cex = 0.7, # Hiển thị hệ số tương quan
    col = col_palette(200),
    title = "Ma trận tương quan các đặc trưng (Pearson)", mar = c(0, 0, 2, 0)
)
dev.off()

# Vẽ hiển thị trực tiếp lên R Viewer
corrplot(cor_matrix,
    method = "color", type = "upper",
    tl.col = "black", tl.srt = 45,
    col = col_palette(200)
)

# --------------------------------------------------------------------=
# 5. PHÂN RÃ CHUỖI THỜI GIAN (TIME-SERIES DECOMPOSITION)
# --------------------------------------------------------------------=
print("⏳ Đang phân rã chuỗi thời gian PM2.5 (Trend, Seasonality, Residuals)...")

# Điền các giá trị NA nhanh bằng forward-fill/backward-fill để đảm bảo
# chuỗi liên tục tuyệt đối, tránh lỗi thuật toán decompose
train_df_filled <- train_df %>% fill(pm25, .direction = "downup")

# Khởi tạo đối tượng Time-Series (ts) với chu kỳ 24 giờ
ts_pm25 <- ts(train_df_filled$pm25, frequency = 24)

# Phân rã chuỗi theo mô hình Cộng (Additive)
decomp <- decompose(ts_pm25, type = "additive")

# Lưu và vẽ biểu đồ phân rã
png(paste0(fig_dir, "pm25_decomposition.png"), width = 1000, height = 800, res = 100)
plot(decomp, col = "darkblue", lwd = 1.2)
dev.off()

# Hiển thị trên R Viewer
plot(decomp, col = "darkblue", lwd = 1.2)

cat(" Đã hoàn thành EDA! Biểu đồ được lưu tại:", fig_dir, "\n")
