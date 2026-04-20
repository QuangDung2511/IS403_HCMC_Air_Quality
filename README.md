# 🌫️ Phân tích và dự báo chất lượng không khí (PM2.5) tại TP.HCM dựa trên Dữ liệu Lịch sử và Cảm biến IoT
### (Real-time Forecasting of Air Quality (PM2.5) in HCMC using Historical Data and IoT Sensors)

> **Lĩnh vực:** Khoa học Dữ liệu (Data Science) & Khoa học Môi trường (Environmental Science)  
> **Môn học:** IS403 - Business Data Analysis | Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM

---

## 👥 Thành viên nhóm

| MSSV | Họ và Tên | Vai trò |
|---|---|---|
| 23520335 | Nguyễn Quang Dũng | Nhóm trưởng |
| 23520322 | Trần Nguyễn Tiến Đức | Thành viên |
| 23520316 | Trần Anh Đức | Thành viên |
| 23520143 | Trần Quốc Bảo | Thành viên |

---

## 1. Tính cấp thiết và Định hướng nghiên cứu

### 1.1. Tính cấp thiết của đề tài
Thành phố Hồ Chí Minh đang đối mặt với những thách thức hệ thống về chất lượng không khí do quá trình đô thị hóa và phát triển công nghiệp nhanh chóng. Nồng độ bụi mịn PM2.5 thường xuyên vượt ngưỡng an toàn, gây rủi ro trực tiếp đến sức khỏe cộng đồng. Hiện nay, các hệ thống quan trắc chủ yếu dừng lại ở việc cung cấp dữ liệu thực tế (Real-time monitoring) mà thiếu đi khả năng dự báo (Forecasting). Sự thiếu hụt các mô hình dự báo ngắn hạn khiến cơ quan quản lý và người dân rơi vào thế bị động trong việc triển khai các biện pháp phòng ngừa và bảo vệ sức khỏe.

### 1.2. Định hướng nghiên cứu
Nghiên cứu này định hướng ứng dụng các kỹ thuật tiên tiến trong Khoa học Dữ liệu và Trí tuệ Nhân tạo để giải quyết bài toán dự báo chuỗi thời gian đa biến (Multivariate Time-Series Forecasting). Để đảm bảo tính toàn diện và độ tin cậy, nghiên cứu triển khai đối sánh và kết hợp ba nhóm phương pháp luận chủ đạo:
- **Mô hình Thống kê (ARIMA, ARIMAX, SARIMA, SARIMAX):** Nắm bắt các quy luật mùa vụ và xu hướng tuyến tính.
- **Mô hình Học máy dựa trên cây (Random Forest, XGBoost, LightGBM):** Khai thác tối đa các đặc trưng khí tượng và đánh giá mức độ quan trọng của các biến số.
- **Mô hình Học sâu (LSTM, GRU):** Mô hình hóa các phụ thuộc thời gian dài hạn và quan hệ phi tuyến phức tạp.
Bằng việc tích hợp đa nguồn dữ liệu từ cảm biến IoT và dữ liệu khí tượng vệ tinh, nghiên cứu hướng tới việc xây dựng một hệ thống có khả năng đưa ra cảnh báo sớm về mức độ ô nhiễm trong 24 giờ tiếp theo.

---

## 2. Mục tiêu nghiên cứu

### 2.1. Mục tiêu tổng quát
Nghiên cứu, thiết kế và xây dựng thành công mô hình dự báo nồng độ bụi mịn PM2.5 tại khu vực TP.HCM trong khung thời gian 24 giờ tới, nhằm hỗ trợ hệ thống cảnh báo sức khỏe cộng đồng.

### 2.2. Mục tiêu cụ thể
- Thu thập và đồng bộ hóa thành công tập dữ liệu lớn về chất lượng không khí và khí tượng tại TP.HCM từ tháng 11/2024 đến nay.
- Phân tích khám phá dữ liệu (EDA) để xác định các quy luật phân bố của PM2.5 theo thời gian (giờ trong ngày, ngày trong tuần, tháng) và sự tương quan giữa các yếu tố thời tiết với sự tích tụ/phân tán của bụi mịn.
- Thiết kế, huấn luyện và so sánh hiệu năng giữa mô hình thống kê truyền thống (SARIMA) và mô hình mạng nơ-ron nhân tạo (LSTM).
- Đề xuất kiến trúc hệ thống cảnh báo sớm, tự động cập nhật dữ liệu từ API để đưa ra dự báo theo thời gian thực.

---

## 3. Đối tượng và Phạm vi nghiên cứu

- **Đối tượng nghiên cứu:** Sự biến động của nồng độ bụi mịn PM2.5 và mối quan hệ tuyến tính/phi tuyến tính của nó với các điều kiện khí tượng.
- **Phạm vi không gian:** Khu vực Thành phố Hồ Chí Minh (lấy tọa độ trung tâm hoặc trạm quan trắc đại diện).
- **Phạm vi thời gian của dữ liệu:** Dữ liệu quá khứ được giới hạn từ tháng 11/2024 đến tháng 02/2026 (đảm bảo tính cập nhật và phản ánh đúng điều kiện kinh tế - xã hội bình thường mới).

---

## 4. Nguồn tư liệu và Dữ liệu nghiên cứu

Để đảm bảo tính khách quan và độ chính xác của mô hình, nghiên cứu sử dụng hai nguồn dữ liệu mở, uy tín và có độ phân giải thời gian theo giờ (hourly):

- **Dữ liệu Chất lượng không khí (OpenAQ):** OpenAQ là nền tảng mã nguồn mở tổng hợp dữ liệu từ các trạm quan trắc IoT trên toàn cầu. Nghiên cứu sẽ trích xuất cột giá trị mục tiêu (Target Variable) là nồng độ pm2.5 (µg/m³), cùng các đặc trưng phụ trợ như pm1 và um003.
- **Dữ liệu Khí tượng (Open-Meteo API):** Thay vì sử dụng dữ liệu tĩnh có sẵn, nghiên cứu tích hợp API của Open-Meteo (dựa trên mô hình ERA5) để lấy dữ liệu khí tượng bề mặt và khí quyển. 8 đặc trưng (features) quan trọng được trích xuất bao gồm:
    - Nhiệt độ bề mặt (`temperature_2m`) và Độ ẩm (`relative_humidity_2m`).
    - Lượng mưa (`precipitation`) và Áp suất (`surface_pressure`).
    - Tốc độ gió (`wind_speed_10m`) và Hướng gió (`wind_direction_10m`).
    - Nhiệt độ ở độ cao 180m (`temperature_180m`) và Độ cao tầng biên (`boundary_layer_height`).

---

## 5. Nội dung và Phương pháp nghiên cứu

Nghiên cứu được tiến hành theo quy trình chuẩn của một dự án Khoa học dữ liệu (**CRISP-DM**), bao gồm các phương pháp cụ thể sau:

### 5.1. Phương pháp thu thập và tiền xử lý dữ liệu (Data Preprocessing)
- **Xử lý giá trị khuyết thiếu (Missing Values Imputation):** Sử dụng phương pháp nội suy tuyến tính (Linear Interpolation) cho các khoảng trống dữ liệu ngắn (dưới 3 giờ) và phương pháp trung bình trượt (Moving Average) cho các khoảng trống lớn hơn. Các biến có tỷ lệ thiếu hụt trên 80% (ví dụ: PM10) sẽ bị loại bỏ để giảm nhiễu.
- **Kỹ thuật trích xuất đặc trưng (Feature Engineering):**
    - Đặc trưng chuỗi thời gian (Lagged Features): Tạo các biến trễ (lag t-1, t-2, ..., t-24) để mô hình học được sự phụ thuộc thời gian.
    - Đặc trưng thời gian tuần hoàn (Cyclical Encoding): Chuyển đổi giờ trong ngày và tháng trong năm thành các hàm sin/cos để biểu diễn tính liên tục của thời gian.
    - Đặc trưng dẫn xuất (Derived Features): Tính toán độ chênh lệch nhiệt độ bề mặt và nhiệt độ 180m để tạo cờ đánh dấu (flag) "Hiện tượng nghịch nhiệt" (Temperature Inversion).

### 5.2. Phương pháp Phân tích Khám phá (EDA)
Sử dụng ngôn ngữ Python (Pandas, Seaborn) hoặc R để trực quan hóa:
- Vẽ biểu đồ hoa gió (Pollution Rose) kết hợp hướng gió và nồng độ bụi để xác định nguồn ô nhiễm lây lan.
- Phân rã chuỗi thời gian (Time-series Decomposition) để tách biệt xu hướng (Trend), tính mùa vụ (Seasonality) và nhiễu (Residuals).

### 5.3. Phương pháp Mô hình hóa (Modeling)
Bài toán được tiếp cận theo hai hướng để đối chiếu:
- **Mô hình SARIMA (Seasonal Autoregressive Integrated Moving Average):** Đại diện cho trường phái thống kê. Phương pháp này đặc biệt hiệu quả trong việc nắm bắt các quy luật lặp lại.
- **Mô hình LSTM (Long Short-Term Memory):** Đại diện cho trường phái Deep Learning. LSTM có kiến trúc cổng (gates) đặc biệt giúp giải quyết vấn đề triệt tiêu đạo hàm, cho phép mạng ghi nhớ các phụ thuộc dài hạn.

### 5.4. Phương pháp Đánh giá (Evaluation Metrics)
Dữ liệu được chia theo tỷ lệ 80/20 (Train/Test) tuần tự theo thời gian. Hiệu suất được đo lường bằng:
- **Root Mean Square Error (RMSE):** Đánh giá mức độ sai số trung bình, phạt nặng các dự báo sai lệch lớn.
- **Mean Absolute Error (MAE):** Phản ánh sai số tuyệt đối trung bình giữa giá trị dự báo và thực tế.

---

## 6. Kết quả dự kiến và Ý nghĩa thực tiễn

- **Về mặt khoa học:** Xây dựng được một bộ pipeline hoàn chỉnh từ khâu gọi API, làm sạch dữ liệu đến huấn luyện mô hình áp dụng riêng cho đặc điểm khí hậu và giao thông của TP.HCM. Cung cấp một nghiên cứu đối chiếu chi tiết giữa mô hình thống kê và mạng nơ-ron nhân tạo trên bộ dữ liệu khí tượng vi mô.
- **Về mặt thực tiễn:** Tạo ra nền tảng cốt lõi cho một ứng dụng "Cảnh báo chất lượng không khí thông minh". Kết quả dự báo 24 giờ tới sẽ giúp các cơ quan quản lý và người dân chủ động sắp xếp lịch trình sinh hoạt ngoài trời, góp phần bảo vệ sức khỏe cộng đồng.

---

## 🚀 Cấu trúc dự án & Pipeline

### Pipeline thực hiện:
`Raw Data` ➡ `Basic Cleaning` ➡ `EDA` ➡ `Preprocessing` ➡ `Feature Engineering` ➡ `Feature Selection` ➡ `Model Training` ➡ `Model Evaluation`

### Các bước chi tiết:
1. **Làm sạch dữ liệu cơ bản:** Gộp dữ liệu từ nhiều nguồn, loại bỏ trùng lặp, sửa kiểu dữ liệu, xử lý thiếu hụt lớn.
2. **Phân tích khám phá (EDA):** Sử dụng R và Python cho phân tích đơn biến, hai biến và đa biến.
3. **Tiền xử lý:** Impute giá trị thiếu, xử lý ngoại lệ, biến đổi phân phối, Encoding, Scaling.
4. **Kỹ thuật đặc trưng:** Lag features, Rolling Window Stats, Temperature Inversion flag.
5. **Lựa chọn đặc trưng:** Feature Importance.
6. **Huấn luyện mô hình:** Chia tập dữ liệu 60/20/20 (Train/Val/Test). Thử nghiệm các dòng: SARIMA, SARIMAX, XGBoost, LightGBM, LSTM, GRU.
7. **Đánh giá & Báo cáo:** Sử dụng RMSE, MAE, MAPE và biểu đồ Actual vs Predicted line chart.

---
<p align="center">
  <em>IS403 - Business Data Analysis | Copyright © 2024</em>
</p>
