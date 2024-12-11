import pandas as pd
import joblib
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import logging

# Cấu hình logging
logging.basicConfig(
    filename='stacking_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Bước 1: Đọc dữ liệu
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # Biến mục tiêu: trust_score
y_test = pd.read_csv('y_test.csv').squeeze()

logging.info("Đọc dữ liệu thành công.")
logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Bước 2: Nạp các base models đã huấn luyện từ trước
svr_model = joblib.load('svr_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('linear_regression_model.pkl')

logging.info("Nạp các base models thành công.")

# Bước 3: Cấu hình Stacking với các Base Models và Final Estimator
estimators = [
    ('svr', svr_model),
    ('rf', rf_model),
    ('lr', lr_model)
]
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    passthrough=False  # Chỉ sử dụng đầu ra của Base Models
)

# Kiểm tra đầu ra của từng base model
logging.info("Kiểm tra đầu ra của từng base model:")
for name, model in estimators:
    base_pred = model.predict(X_test)
    logging.info(f"{name} Predictions (5 values): {base_pred[:5]}")

# Bước 4: Huấn luyện mô hình Stacking
logging.info("Bắt đầu huấn luyện mô hình Stacking...")
stacking_model.fit(X_train, y_train)
logging.info("Hoàn thành huấn luyện mô hình Stacking.")

# Bước 5: Cross-Validation để đánh giá tổng quát
cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='r2')
mean_cv_score = np.mean(cv_scores)
logging.info(f"Cross-Validation R2 Scores: {cv_scores}")
logging.info(f"Mean Cross-Validation R2 Score: {mean_cv_score}")

# Bước 6: Đánh giá trên tập kiểm tra
y_pred = stacking_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Đánh Giá Mô Hình Stacking ---")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

logging.info(f"Đánh giá trên tập kiểm tra: MSE={mse}, RMSE={rmse}, MAE={mae}, R2={r2}")

# Lưu kết quả dự đoán và chỉ số đánh giá
results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
results_df.to_csv("stacking_predictions.csv", index=False)

metrics = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("stacking_metrics.csv", index=False)

logging.info("Lưu dự đoán và chỉ số đánh giá thành công.")

# Bước 7: Lưu mô hình Stacking đã huấn luyện
stacking_model_filename = "stacking_model_final.pkl"
joblib.dump(stacking_model, stacking_model_filename)
logging.info(f"Mô hình Stacking đã được lưu vào file: {stacking_model_filename}")

# In thông báo hoàn tất
print(f"\nMô hình Stacking đã được lưu vào file: {stacking_model_filename}")
