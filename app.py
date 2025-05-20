import os
import sys
import joblib
import streamlit as st
import sklearn
import numpy as np
import pandas as pd

# 设置页面配置
st.set_page_config(page_title="Lung Nodule Risk Prediction", page_icon="🫁", layout="wide")

# 修改模型加载函数以适应 Streamlit 部署
def safe_load_model(filename):
    try:
        # 使用 st.secrets 或相对路径
        possible_paths = [
            os.path.join(os.getcwd(), filename),  # 当前工作目录
            os.path.join(os.path.dirname(__file__), filename),  # 脚本所在目录
            filename  # 直接文件名
        ]
        
        for path in possible_paths:
            st.write(f"Trying to load from: {path}")
            if os.path.exists(path):
                model = joblib.load(path)
                st.write(f"Successfully loaded {filename} from {path}")
                return model
        
        st.error(f"Model file {filename} not found in any of the expected locations")
        return None
    
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

# 尝试加载模型和特征
try:
    model_8mm = safe_load_model('GBC_8mm_model.joblib')
    model_30mm = safe_load_model('GBC_30mm_model.joblib')
    features_8mm = safe_load_model('GBC_8mm_features.joblib')
    features_30mm = safe_load_model('GBC_30mm_features.joblib')
except Exception as e:
    st.error(f"Unexpected error during model loading: {e}")
    model_8mm = model_30mm = features_8mm = features_30mm = None

# 检查模型是否成功加载
if model_8mm is None or model_30mm is None or features_8mm is None or features_30mm is None:
    st.error("Failed to load one or more model files. Please check the files.")
    st.stop()

# 创建输入表单函数
def get_user_input(features, nodule_diameter):
    input_data = {}
    
    # 将特征平均分配到两列
    mid = len(features) // 2
    col1_features = features[:mid]
    col2_features = features[mid:]
    
    # 创建两列
    col1, col2 = st.columns(2)
    
    # 输入处理
    def process_features(features_list, column):
        for feature in features_list:
            if feature == 'Nodule diameter':
                column.write(f"Nodule Diameter (mm): {nodule_diameter}")
                input_data[feature] = nodule_diameter
            elif feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE', 'ProGRP']:
                input_data[feature] = column.number_input(
                    f"Enter {feature}", 
                    min_value=0.0, 
                    step=0.1, 
                    key=f"{feature}_{column}"
                )
            else:
                input_data[feature] = column.selectbox(
                    f"Select {feature}", 
                    [0, 1], 
                    format_func=lambda x, f=feature: 
                        "Female" if f == 'Gender' and x == 0 else 
                        "Male" if f == 'Gender' and x == 1 else 
                        "No" if x == 0 else "Yes",
                    key=f"{feature}_{column}"
                )
    
    # 处理第一列和第二列
    with col1:
        process_features(col1_features, col1)
    
    with col2:
        process_features(col2_features, col2)
    
    return input_data

# 预测函数
def predict_risk(input_data, model, features):
    # 确保按照原始特征顺序创建 DataFrame
    input_df = pd.DataFrame([{feature: input_data[feature] for feature in features}])
    
    # 预测
    prediction = model.predict_proba(input_df)
    malignancy_prob = prediction[0][1]

    return malignancy_prob

# 主函数
def main():
    st.title("🫁 TEB Lung Nodule Malignancy Risk Predictor")
    
    with st.sidebar:
        st.header("Model Selection")
        nodule_diameter = st.number_input("Nodule Diameter (mm)", min_value=0.0, max_value=50.0, step=0.1)
        
        if nodule_diameter <= 8:
            st.info("8mm Model Selected")
            features = features_8mm
            model = model_8mm
        elif nodule_diameter <= 30:
            st.info("30mm Model Selected")
            features = features_30mm
            model = model_30mm
        else:
            st.error("Nodule diameter out of predictive range")
            return
    
    input_data = get_user_input(features, nodule_diameter)
    
    if st.sidebar.button("Predict Risk"):
        # 预测风险
        malignancy_prob = predict_risk(input_data, model, features)
        
        # 结果展示
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown("### Prediction Results")
            st.metric("Malignancy Risk", f"{malignancy_prob:.2%}")
        
        with result_col2:
            st.markdown("### Risk Interpretation")
            if malignancy_prob < 0.2:
                st.success("Low Risk: Close monitoring recommended")
            elif malignancy_prob < 0.5:
                st.warning("Moderate Risk: Further investigation suggested")
            else:
                st.error("High Risk: Immediate clinical consultation advised")

# 自定义 CSS 样式
st.markdown("""
<style>
.main-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.stNumberInput > div > div > input {
    background-color: white;
    border: 1px solid #4a84c4;
    border-radius: 5px;
}
.stSelectbox > div > div > select {
    background-color: white;
    border: 1px solid #4a84c4;
    border-radius: 5px;
}
.stButton > button {
    background-color: #4a84c4;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
}
.stMetric > div {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.stMetric > div > div {
    color: #4a84c4;
}
</style>
""", unsafe_allow_html=True)

# 运行主函数
if __name__ == "__main__":
    main()
