# Danh sách Công việc Dự án Dự báo Chất lượng Không khí (HCMC PM2.5)

Sau khi hoàn thành giai đoạn Feature Engineering, đây là lộ trình tiếp theo để triển khai mô hình và hoàn thiện báo cáo.

## Giai đoạn 2: Lựa chọn Đặc trưng & Tiền xử lý (Tiếp theo)

### 1. Lựa chọn đặc trưng (Feature Selection)
- **Mục tiêu**: Loại bỏ các biến không cần thiết hoặc gây nhiễu để tối ưu hóa hiệu suất mô hình.
- **Các biến cần loại bỏ ngay (Dựa trên EDA)**:
    - `pm1`, `um003` (Trùng lặp thông tin với PM2.5).
    - `temperature (OpenAQ)`, `relativehumidity (OpenAQ)` (Dữ liệu từ OpenAQ có độ trễ/sai số so với cảm biến thực tế).
    - `wind_direction_10m` (Đã được thay thế bằng vector hóa `wind_u` và `wind_v`).
- **Kỹ thuật đề xuất**: Sử dụng **Random Forest Feature Importance** hoặc **XGBoost Importance** để xếp hạng và giữ lại Top 20-30 đặc trưng có ảnh hưởng lớn nhất.

### 2. Tiền xử lý dữ liệu (Preprocessing Pipelines)
Chia làm 2 luồng xử lý riêng biệt cho 2 nhóm mô hình:
- **Nhóm 1 (Tree-based Models - XGBoost, LightGBM)**: 
    - Thực hiện Label Encoding hoặc Frequency Encoding cho các biến phân loại (như `wind_condition`). 
    - Giữ nguyên thang đo cho các biến số (Tree models không nhạy cảm với scale).
- **Nhóm 2 (Statistical/DL Models - ARIMA, LSTM, GRU)**:
    - **Transform**: Thực hiện Log-transform hoặc Box-Cox cho PM2.5 để giảm độ lệch (skewness).
    - **Scaling**: Thực hiện **StandardScaler** hoặc **MinMaxScaler** để đưa dữ liệu về cùng một khoảng (giúp hội tụ nhanh hơn).

---

## Giai đoạn 3: Huấn luyện & Đánh giá Mô hình (06/04 – 14/04)

- **Chia Dữ liệu (Time-series Split)**: 
    - Chia theo trình tự thời gian (Không dùng Shuffle).
    - Tỷ lệ: **60% Train / 20% Validation / 20% Test**.
- **Huấn luyện các Mô hình**:
    - **Mô hình Thống kê**: ARIMA, SARIMA, ARIMAX, SARIMAX (Sử dụng các biến ngoại sinh từ weather).
    - **Mô hình Học máy/Học sâu**: XGBoost, LightGBM, LSTM, GRU.
- **Đánh giá Hiệu suất**:
    - Chỉ số đo lường: **RMSE, MAE và MAPE**.
    - Hình ảnh hóa: Vẽ biểu đồ **Actual vs Predicted** trên tập Test để so sánh trực quan.

---

## Giai đoạn 4: Viết Báo cáo & Đào tạo chéo (15/04 – 21/04)

- **Phân tích Chuyên sâu (In-depth Analysis)**: 
    - Giải thích rõ yếu tố thời tiết nào (Nhiệt độ, Độ ẩm, hay Hướng gió) tác động mạnh nhất đến PM2.5 tại TP.HCM.
    - So sánh ưu/nhược điểm giữa nhóm mô hình Thống kê và Học sâu trong bài toán này.
- **Trích dẫn Tài liệu**: 
    - Tìm và tích hợp các trích dẫn từ bài báo khoa học về môi trường (Scopus/ISI) để giải thích hiện tượng (ví dụ: hiệu ứng "washout" của mưa).
- **Phân công & Chuyển giao Kiến thức (Cross-training)**:
    - Tạo bảng phân công chi tiết đóng góp của từng thành viên.
    - **Bắt buộc**: Mỗi thành viên phải thuyết trình/hướng dẫn lại phần việc của mình cho cả nhóm. 
    - Đảm bảo 100% thành viên có thể trả lời các câu hỏi phản biện (Q&A) về bất kỳ phần nào của dự án.
