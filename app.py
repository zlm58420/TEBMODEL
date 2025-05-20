import os
import sys
import joblib
import streamlit as st
import sklearn
import numpy as np
import packaging.version
import pandas as pd
import matplotlib.pyplot as plt

# 将 set_page_config 移到最前面
st.set_page_config(page_title="Lung Nodule Risk Prediction", page_icon="🫁", layout="wide")

# 打印详细的系统和环境信息
st.write("Python Version:", sys.version)
st.write("scikit-learn Version:", sklearn.__version__)
st.write("joblib Version:", joblib.__version__)

# 打印当前目录和文件列表
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# 尝试导入SHAP，如果失败则提供警告
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    st.warning("SHAP library not available. Some visualizations will be limited.")
    SHAP_AVAILABLE = False

# 安全的模型加载函数
def safe_load_model(filename):
    try:
        # 尝试使用完整路径加载
        full_path = os.path.join(os.getcwd(), filename)
        model = joblib.load(full_path)
        st.write(f"Successfully loaded {filename} from {full_path}")
        return model
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        st.error(f"Full error details: {sys.exc_info()}")
        return None

# 加载模型和特征
model_8mm = safe_load_model('GBC_8mm_model.joblib')
model_30mm = safe_load_model('GBC_30mm_model.joblib')
features_8mm = safe_load_model('GBC_8mm_features.joblib')
features_30mm = safe_load_model('GBC_30mm_features.joblib')

# 检查模型是否成功加载
if model_8mm is None or model_30mm is None or features_8mm is None or features_30mm is None:
    st.error("Failed to load one or more model files. Please check the files.")
    st.stop()

# 版本兼容性处理
def patch_numpy_version():
    try:
        import numba
        current_numpy_version = packaging.version.parse(np.__version__)
        if current_numpy_version > packaging.version.parse('2.0'):
            import numba.__init__
            def _ensure_critical_deps():
                pass
            numba.__init__._ensure_critical_deps = _ensure_critical_deps
    except ImportError:
        pass

patch_numpy_version()

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

# 预测和解释函数
def predict_and_explain(input_data, model, features):
    # 确保按照原始特征顺序创建 DataFrame
    input_df = pd.DataFrame([{feature: input_data[feature] for feature in features}])
    
    # 预测
    prediction = model.predict_proba(input_df)
    malignancy_prob = prediction[0][1]

    # SHAP 解释
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # 处理 SHAP 值的索引问题
            shap_plot_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")
            return malignancy_prob, None
    else:
        return malignancy_prob, None

    return malignancy_prob, shap_plot_values

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
        malignancy_prob, shap_plot_values = predict_and_explain(input_data, model, features)
        
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
        
        # SHAP 可视化
        if SHAP_AVAILABLE and shap_plot_values is not None:
            st.markdown("### Feature Impact Analysis")
            col_shap1, col_shap2 = st.columns(2)
            
            with col_shap1:
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_plot_values, input_df, plot_type="bar", show=False)
                st.pyplot(fig)
            
            with col_shap2:
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_plot_values, input_df, show=False)
                st.pyplot(fig)

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
