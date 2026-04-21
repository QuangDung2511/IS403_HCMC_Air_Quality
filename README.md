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
- Thu thập và đồng bộ hóa thành công tập dữ liệu lớn về chất lượng không khí và khí tượng tại TP.HCM từ tháng 11/2024 đến tháng 2/2026.
- Phân tích khám phá dữ liệu (EDA) để xác định các quy luật phân bố của PM2.5 theo thời gian (giờ trong ngày, ngày trong tuần, tháng) và sự tương quan giữa các yếu tố thời tiết với sự tích tụ/phân tán của bụi mịn.
- Thiết kế, huấn luyện và so sánh hiệu năng giữa mô hình thống kê truyền thống và mô hình mạng nơ-ron nhân tạo.
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

Nghiên cứu được tiến hành theo quy trình chuẩn của một dự án Khoa học dữ liệu (**CRISP-DM**), theo chuỗi pipeline cốt lõi:
`Raw Data` ➡ `Basic Cleaning` ➡ `EDA` ➡ `Preprocessing` ➡ `Model Training` ➡ `Model Evaluation`

### 5.1. Dữ liệu thô và Làm sạch cơ bản (Raw Data & Basic Cleaning)
- Thu thập và gộp dữ liệu không gian, thời gian từ các trạm quan trắc (OpenAQ) và tham số khí tượng (Open-Meteo API).
- Căn chỉnh mốc thời gian (timezone), loại bỏ các bản ghi trùng lặp và loại bỏ các biến có tỷ lệ thiếu hụt quá cao (trên 80%).

### 5.2. Phân tích Khám phá dữ liệu (EDA - Train Only)
- Để bảo vệ tính bảo mật của dữ liệu tương lai và không gây rò rỉ dữ liệu (Data Leakage), toàn bộ quá trình phân tích khám phá (EDA) **chỉ được tiến hành trên tập Huấn luyện (Train Data)**.
- Đánh giá phân phối của PM2.5, tìm tương quan đa biến giữa điều kiện thời tiết (Pollution Rose, Heatmap) và phân rã các tính chất thành phần chuỗi thời gian (STL Decomposition).

### 5.3. Tiền xử lý dữ liệu (Preprocessing)
Với đặc thù chuỗi thời gian đa biến phức tạp, giai đoạn tiền xử lý (Preprocessing) quyết định phần lớn sự thành công của dự án và được thực hiện một cách cực kỳ chi tiết:
- **Xử lý giá trị khuyết thiếu (Imputation):** Áp dụng nội suy tuyến tính (Linear Interpolation) cho các khoảng rỗng ngắn dưới 3 giờ. Đối với các khoảng trống rỗng lớn hơn sẽ được khôi phục dựa trên phương pháp trung bình trượt kết hợp giá trị quy nạp.
- **Kỹ thuật trích xuất đặc trưng (Feature Engineering):** Tạo dựng các trễ thời gian (Lag features: t-1, t-2, t-24) giúp mô hình ghi nhớ giá trị quá khứ, biến đổi Cyclical Encoding hàm lượng giác (Sin/Cos) cho các chu kỳ tuần hoàn (giờ trong ngày, tháng trong năm), cùng với tính toán "cờ" (flag) nghịch nhiệt độ bức xạ bề mặt.
- **Biến đổi mục tiêu (Log1p Transformation):** Vi PM2.5 có thiết dạng phân phối lệch phải (Right-skewed) với nhiều sự kiện ô nhiễm cao bất thường, kỹ thuật log được dùng để ép phân phối về tiệm cận chuẩn, ổn định phương sai cho các mô hình DL.
- **Xử lý ngoại lệ (Capping/Outliers Handling):** Cắt định mức các biên ngoại lệ theo percentile (1% – 99%) để giảm thiểu sự rung lắc học của mạng nơ-ron mà không mất đi gai tính hiệu.
- **Chuẩn hóa tỷ lệ (Scaling):** Sử dụng các Scaler (MinMaxScaler) chuyên dụng trên các biến khí hậu phi tuyến để thống nhất thang đo trọng số (weights) áp dụng cho Deep Learning.

### 5.4. Huấn luyện mô hình (Model Training)
- Chia tách tập train/val/test tuần tự và áp dụng `TimeSeriesSplit` để đối sánh đa cực.
- Triển khai từ các Baseline mô hình Thống kê (ARIMA, ARIMAX, SARIMA, SARIMAX), nhóm dạng Cây (Random Forest, XGBoost, LightGBM) cho tới nhóm Nơ-ron (LSTM, GRU). Nổi bật hệ thống còn đào tạo phiên bản mô hình **Hybrid (SARIMAX kết hợp GRU)** để triệt tiêu các đặc trưng tuyến tính lẫn phi tuyến tính.

### 5.5. Đánh giá mô hình (Model Evaluation)
- Căn chỉnh dựa trên chỉ số RMSE, MAE, MAPE để quyết định sai số tương đối và độ lệch tuyệt đối. Đánh vần năng suất cảnh báo ở các khung giờ cao điểm có PM2.5 tăng vọt đột biến.

---

## 6. Kết quả dự kiến và Ý nghĩa thực tiễn

- **Về mặt khoa học:** Xây dựng được một bộ pipeline hoàn chỉnh từ khâu gọi API, làm sạch dữ liệu đến huấn luyện mô hình áp dụng riêng cho đặc điểm khí hậu và giao thông của TP.HCM. Cung cấp một nghiên cứu đối chiếu chi tiết giữa mô hình thống kê và mạng nơ-ron nhân tạo trên bộ dữ liệu khí tượng vi mô.
- **Về mặt thực tiễn:** Tạo ra nền tảng cốt lõi cho một ứng dụng "Cảnh báo chất lượng không khí thông minh". Kết quả dự báo 24 giờ tới sẽ giúp các cơ quan quản lý và người dân chủ động sắp xếp lịch trình sinh hoạt ngoài trời, góp phần bảo vệ sức khỏe cộng đồng.

## 📝 Biên soạn Báo cáo (LaTeX Report)

Báo cáo chi tiết của đồ án (Đồ án IS403) được trình bày trong thư mục `report/`.

### Yêu cầu hệ thống:
1.  **TeX Live** hoặc **MiKTeX** (đã được cài đặt).
2.  **Perl**: Yêu cầu để chạy `latexmk`. (Đã cài đặt **Strawberry Perl**).

### Cách biên soạn:
-   **Cách 1 (Tùy chọn khuyến nghị):** Mở `report/main.tex` bằng VS Code (với extension LaTeX Workshop) hoặc TeXstudio. `latexmk` sẽ tự động biên soạn.
-   **Cách 2 (Thủ công - Windows):** Chạy script PowerShell được cung cấp:
    ```powershell
    cd report
    .\compile_report.ps1
    ```

Chi tiết cài đặt bổ sung có thể xem tại [INSTALL_PERL.md](report/INSTALL_PERL.md).

---
<p align="center">
  <em>IS403 - Business Data Analysis | Copyright © 2024</em>
</p>
