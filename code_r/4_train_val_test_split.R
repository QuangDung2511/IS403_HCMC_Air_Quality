# =====================================================================
# BƯỚC 4: CHIA TẬP DỮ LIỆU (Train / Validation / Test Split)
# Lưu ý: Với Time-Series, dữ liệu được cắt theo đúng trình tự thời gian
# =====================================================================

# Nạp thư viện
library(tidyverse)
library(lubridate)

# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN
base_dir <- "C:/Users/TRAN ANH DUC/OneDrive/Máy tính/IS403_HCMC_Air_Quality"
path_input <- paste0(base_dir, "/data/processed_R/hcmc_merged_cleaned.csv")
out_dir <- paste0(base_dir, "/data/processed_R/")

# 2. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
df <- read_csv(path_input, show_col_types = FALSE)

# Đảm bảo dữ liệu được sắp xếp chuẩn xác theo trục thời gian trước khi cắt
df <- df %>%
    mutate(datetime_local = ymd_hms(datetime_local)) %>%
    arrange(datetime_local)

# 3. TÍNH TOÁN KÍCH THƯỚC CÁC TẬP (60% Train - 20% Val - 20% Test)
n_total <- nrow(df)
train_size <- floor(n_total * 0.60)
val_size <- floor(n_total * 0.20)
test_size <- n_total - train_size - val_size

# 4. CẮT DỮ LIỆU (Slicing)
train_df <- df[1:train_size, ]
val_df <- df[(train_size + 1):(train_size + val_size), ]
test_df <- df[(train_size + val_size + 1):n_total, ]

# 5. HIỂN THỊ THÔNG TIN ĐỂ KIỂM TRA (Sanity Check)
cat("Tổng số dòng toàn bộ dữ liệu:", n_total, "\n\n")

cat("Kích thước tập Train (60%):", nrow(train_df), "dòng\n")
cat("   -> Từ", as.character(min(train_df$datetime_local)), "đến", as.character(max(train_df$datetime_local)), "\n\n")

cat("Kích thước tập Val   (20%):", nrow(val_df), "dòng\n")
cat("   -> Từ", as.character(min(val_df$datetime_local)), "đến", as.character(max(val_df$datetime_local)), "\n\n")

cat("Kích thước tập Test  (20%):", nrow(test_df), "dòng\n")
cat("   -> Từ", as.character(min(test_df$datetime_local)), "đến", as.character(max(test_df$datetime_local)), "\n\n")

# 6. LƯU KẾT QUẢ VÀO THƯ MỤC R
write_csv(train_df, paste0(out_dir, "train_split.csv"))
write_csv(val_df, paste0(out_dir, "val_split.csv"))
write_csv(test_df, paste0(out_dir, "test_split.csv"))

cat(" Đã chia tập dữ liệu thành công! File lưu tại:", out_dir, "\n")
