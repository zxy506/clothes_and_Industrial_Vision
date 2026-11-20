"""Run with: streamlit run app.py"""

import streamlit as st

from src.services.generation_service import load_generator
from src.services.inference_service import load_classifier
from src.services.fashion_inference_service import load_classifier as load_fashion_classifier
from ui.pages import experiments, fashion_recognition, generation, history, home, recognition, settings


st.set_page_config(page_title="Industrial Vision", page_icon="🔬", layout="wide")
st.markdown("""<style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    h1, h2, h3 {color: #16324f;}
    [data-testid="stSidebar"] {background-color: #eef3f7;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="加载分类模型中…")
def cached_classifier(mode, seed, checkpoint_mtime):
    return load_classifier(mode, seed)


@st.cache_resource(show_spinner="加载衣物分类模型中…")
def cached_fashion_classifier(mode, seed, checkpoint_mtime):
    return load_fashion_classifier(mode, seed)


@st.cache_resource(show_spinner="加载生成模型中…")
def cached_generator(checkpoint_mtime):
    return load_generator()


pages = ["首页", "工业缺陷识别", "衣物识别", "CVAE 样本生成", "实验结果对比", "历史检测记录", "系统设置"]
st.sidebar.title("Industrial Vision")
st.sidebar.caption("工业表面缺陷分类 · 本地运行")
selected = st.sidebar.radio("导航", pages)
st.sidebar.divider()
st.sidebar.caption("图像级分类研究系统 · NEU-CLS 与 Fashion-MNIST")

if selected == "首页":
    home.render()
elif selected == "工业缺陷识别":
    recognition.render(cached_classifier)
elif selected == "衣物识别":
    fashion_recognition.render(cached_fashion_classifier)
elif selected == "CVAE 样本生成":
    generation.render(cached_generator)
elif selected == "实验结果对比":
    experiments.render()
elif selected == "历史检测记录":
    history.render()
else:
    settings.render()
