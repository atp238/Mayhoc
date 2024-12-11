import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import StackingRegressor  # Thêm import này để đảm bảo load model được

# Kiểm tra và load mô hình
try:
    model_filename = "stacking_model_final.pkl"
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Không tìm thấy file model. Vui lòng kiểm tra lại đường dẫn tới file model.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi load model: {str(e)}")
    st.stop()

# Khởi tạo giao diện chatbot-like
st.title("Đánh Giá Độ Uy Tín Dự Án DeFi")
st.write("Nhập các dữ liệu đã chuẩn hóa để mô hình đánh giá.")

# Hiển thị bảng nhập liệu
st.write("### Nhập Dữ liệu Dự Án Trong Bảng:")

data_columns = [
    "Market Cap (chuẩn hóa)",
    "Current TVL (chuẩn hóa)",
    "TVL Growth (1 month, chuẩn hóa)",
    "TVL Stability (chuẩn hóa)",
    "TVL/MCap Ratio (chuẩn hóa)",
    "Project Age (chuẩn hóa)",
    "Funding (chuẩn hóa)"
]

# Khởi tạo bảng dữ liệu với các giá trị mặc định
default_data = {
    "Market Cap (chuẩn hóa)": [0.0],
    "Current TVL (chuẩn hóa)": [0.0],
    "TVL Growth (1 month, chuẩn hóa)": [0.0],
    "TVL Stability (chuẩn hóa)": [0.0],
    "TVL/MCap Ratio (chuẩn hóa)": [0.0],
    "Project Age (chuẩn hóa)": [0.0],
    "Funding (chuẩn hóa)": [0.0]
}

# Sử dụng data_editor với validation
input_data = st.data_editor(
    pd.DataFrame(default_data),
    num_rows="dynamic",
    use_container_width=True,
    disabled=False,
    hide_index=True,
)

# Hiển thị nút để thực hiện dự đoán
if st.button("Đánh Giá"):
    try:
        # Kiểm tra dữ liệu đầu vào
        if input_data.isnull().values.any():
            st.error("Vui lòng điền đầy đủ tất cả các trường dữ liệu!")
            st.stop()
            
        # Kiểm tra giá trị nằm trong khoảng [0,1]
        if (input_data < 0).values.any() or (input_data > 1).values.any():
            st.error("Tất cả các giá trị phải nằm trong khoảng từ 0 đến 1!")
            st.stop()

        # Chuyển đổi bảng dữ liệu thành DataFrame để dự đoán
        processed_data = pd.DataFrame(input_data)

        # Dự đoán kết quả Trust Score
        predictions = model.predict(processed_data)
        trust_score = max(0, min(100, predictions[0] * 100))  # Giới hạn trong khoảng [0,100]

        # Hiển thị kết quả
        st.write("### Kết Quả Đánh Giá")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.progress(int(trust_score))
        with col2:
            st.metric("Độ Uy Tín", f"{trust_score:.2f}%")

        # Hiển thị radar chart phân tích tương quan
        st.write("### Phân Tích Tương Quan Các Yếu Tố Ảnh Hưởng")
        labels = [
            "Market Cap",
            "Current TVL",
            "TVL Growth (1m)",
            "TVL Stability",
            "TVL/MCap Ratio",
            "Project Age",
            "Funding"
        ]
        values = processed_data.iloc[0].values
        avg_values = [0.5] * len(labels)  # Giá trị trung bình

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name='Dự Án Hiện Tại',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=labels,
            fill='toself',
            name='Giá Trị Trung Bình',
            line=dict(color='#ff7f0e')
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=10)
                )
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=80, r=80, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi trong quá trình đánh giá: {str(e)}")

# Thêm phần giải thích về các chỉ số
with st.expander("Giải thích các chỉ số"):
    st.markdown("""
    - **Market Cap**: Vốn hóa thị trường của dự án
    - **Current TVL**: Tổng giá trị khóa hiện tại trong dự án
    - **TVL Growth**: Tốc độ tăng trưởng TVL trong 1 tháng
    - **TVL Stability**: Độ ổn định của TVL
    - **TVL/MCap Ratio**: Tỷ lệ giữa TVL và vốn hóa thị trường
    - **Project Age**: Tuổi đời của dự án
    - **Funding**: Vốn đầu tư nhận được
    
    *Lưu ý: Tất cả các giá trị đầu vào đều đã được chuẩn hóa về khoảng [0,1]*
    """)

# Thêm footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #666666; font-size: 0.8em;'>
            Developed by DeFi Trust Score Team | 2024
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)