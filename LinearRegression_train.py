import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# Bước 1: Đọc các tệp dữ liệu huấn luyện và kiểm tra đã có sẵn
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # squeeze() để chuyển thành Series
y_test = pd.read_csv('y_test.csv').squeeze()

# Đảm bảo chỉ sử dụng các thuộc tính cần thiết (giả sử các thuộc tính đã chuẩn hóa)
features = ["market_cap", "tvl_current", "tvl_growth_1m", "tvl_stability",
            "tvl_mcap_ratio", "project_age", "funding"]

X_train = X_train[features]
X_test = X_test[features]

# Bước 2: Khởi tạo và huấn luyện mô hình Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Bước 3: Đánh giá mô hình trên tập kiểm tra
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Đánh Giá Mô Hình Linear Regression ---")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

# Chuyển đổi y_pred sang tỷ lệ phần trăm và in kết quả
y_pred_percentage = y_pred * 100
print("\n5 Giá Trị Dự Đoán Đầu Tiên (Tỷ Lệ Phần Trăm):", y_pred_percentage[:5])

# Bước 4: Lưu mô hình đã huấn luyện
model_filename = "linear_regression_model.pkl"
joblib.dump(lr_model, model_filename)
print(f"\nMô hình Linear Regression đã được lưu vào file: {model_filename}")

# Lưu các tham số của mô hình
params_filename = "linear_regression_params.pkl"
params = {'coef': lr_model.coef_, 'intercept': lr_model.intercept_}
joblib.dump(params, params_filename)
print(f"Các tham số của mô hình đã được lưu vào file: {params_filename}")
