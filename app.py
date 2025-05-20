import os
import sys
import joblib
import streamlit as st
import sklearn
import numpy as np
import pandas as pd

# 诊断性输出
st.write("Python Version:", sys.version)
st.write("Scikit-learn Version:", sklearn.__version__)

# 安全的模型加载函数
def safe_load_model(filename):
    # 尝试多种加载方式
    possible_paths = [
        os.path.join(os.getcwd(), filename),
        filename
    ]
    
    for path in possible_paths:
        st.write(f"尝试从 {path} 加载")
        
        if os.path.exists(path):
            try:
                # 使用更安全的加载方式
                with open(path, 'rb') as f:
                    model = joblib.load(f)
                st.write(f"成功加载 {filename}")
                return model
            except Exception as e:
                st.error(f"加载 {path} 失败: {e}")
    
    st.error(f"未找到模型文件 {filename}")
    return None

# 加载模型的函数
def load_all_models():
    try:
        model_8mm = safe_load_model('GBC_8mm_model.joblib')
        model_30mm = safe_load_model('GBC_30mm_model.joblib')
        features_8mm = safe_load_model('GBC_8mm_features.joblib')
        features_30mm = safe_load_model('GBC_30mm_features.joblib')
        
        return model_8mm, model_30mm, features_8mm, features_30mm
    
    except Exception as e:
        st.error(f"加载模型时发生错误: {e}")
        return None, None, None, None

# 加载模型
model_8mm, model_30mm, features_8mm, features_30mm = load_all_models()

# 检查模型是否成功加载
if model_8mm is None or model_30mm is None or features_8mm is None or features_30mm is None:
    st.error("模型加载失败，请检查文件")
    st.stop()

# 后续代码保持不变...

# 后续的函数保持不变
def get_user_input(features, nodule_diameter):
    # ... (保持原有代码不变)

def predict_risk(input_data, model, features):
    # ... (保持原有代码不变)

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
        try:
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
        
        except Exception as e:
            st.error(f"预测过程中发生错误: {e}")


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
