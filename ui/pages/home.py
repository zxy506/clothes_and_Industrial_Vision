import json
from pathlib import Path

import streamlit as st

from src.services.inference_service import ROOT


def render():
    st.title("Industrial Vision")
    st.subheader("工业表面缺陷与衣物图像识别平台")
    st.write("提供 NEU-CLS 六类工业缺陷识别、Fashion-MNIST 十类衣物识别和工业 CVAE 样本生成。两套分类模型独立训练。")
    st.info("识别结果是模型预测，需结合人工检查用于实际工业判断。")
    left, right = st.columns(2)
    with left:
        st.markdown("### 数据与模型")
        st.markdown("**NEU-CLS：** 六类钢材表面缺陷，主实验每类使用 90 张真实训练图像。")
        st.markdown("**Fashion-MNIST：** 十分类课程对照实验，与工业模型独立训练。")
        st.markdown("**算法：** 自建 CNN 分类器与条件 VAE 生成器。")
    with right:
        st.markdown("### 运行状态")
        count = len(list((ROOT / "checkpoints" / "neu").glob("cnn_?_seed?.pt")))
        st.metric("可用 NEU 分类权重", f"{count}/9")
        st.metric("NEU 生成器", "已就绪" if (ROOT / "checkpoints/neu/cvae_seed0.pt").exists() else "缺失")
        fashion_count = sum((ROOT / f"checkpoints/fashion/cnn_{mode}_seed{seed}_30epochs.pt").exists()
                            for mode in ("A", "B", "C25", "C") for seed in (0, 1, 2))
        st.metric("可用 Fashion 分类权重", f"{fashion_count}/12")
        st.caption("全部模型与推理均在本机运行，不调用外部 API。")
    st.markdown("### 实验摘要")
    for name, file in (("课程实验 Fashion-MNIST", "fashion_final.json"),
                       ("工业实验 NEU-CLS", "neu_final.json")):
        path = ROOT / "results" / "summary" / file
        if not path.exists():
            st.warning(f"{name}：结果文件缺失")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))["summary"]
        st.write(f"**{name}**")
        cols = st.columns(3)
        for col, mode in zip(cols, ("A", "B", "C")):
            col.metric(mode, f"{data[mode]['clean_accuracy_mean' if file.startswith('fashion') else 'accuracy_mean']:.2%}")
    st.caption("摘要数值来自正式实验结果文件；更多指标见“实验结果对比”。")
