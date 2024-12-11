import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# Bước 1: Đọc các tệp dữ liệu huấn luyện và kiểm tra đã có sẵn
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # squeeze() để chuyển thành Series
y_test = pd.read_csv('y_test.csv').squeeze()

# Bước 2: Khởi tạo và huấn luyện mô hình SVR với Grid Search và Cross-Validation
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'kernel': ['linear', 'rbf']
}
svr_model = SVR()
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='r2')
print("Bắt đầu huấn luyện mô hình với Grid Search...")
grid_search.fit(X_train, y_train)
print("Hoàn thành Grid Search!")

# Lấy mô hình tốt nhất và tham số tốt nhất
best_svr_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Kiểm tra hiệu suất của mô hình bằng Cross-Validation
cross_val_scores = cross_val_score(best_svr_model, X_train, y_train, cv=5, scoring='r2')
mean_cross_val_score = np.mean(cross_val_scores)

# Bước 3: Đánh giá mô hình trên tập kiểm tra
y_pred = best_svr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Đánh Giá Mô Hình ---")
print("Best Parameters from Grid Search:", best_params)
print("Mean Cross-Validation R2 Score:", mean_cross_val_score)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

# Chuyển đổi y_pred sang tỷ lệ phần trăm và in kết quả (nếu cần)
y_pred_percentage = y_pred * 100
print("\n5 Giá Trị Dự Đoán Đầu Tiên (Tỷ Lệ Phần Trăm):", y_pred_percentage[:5])

# Bước 4: Lưu mô hình đã huấn luyện và tham số tốt nhất
model_filename = "svr_model_final.pkl"
joblib.dump(best_svr_model, model_filename)
print(f"\nMô hình đã được lưu vào file: {model_filename}")

params_filename = "smo_model_params.pkl"
joblib.dump(best_params, params_filename)
print(f"Tham số tốt nhất đã được lưu vào file: {params_filename}")
