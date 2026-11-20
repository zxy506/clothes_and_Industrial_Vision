import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.services.inference_service import ROOT


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _confusion(dataset: str, mode: str, classes):
    records = []
    for seed in (0, 1, 2):
        path = ROOT / "results" / "raw" / f"{dataset}_final" / f"{mode}_seed{seed}.json"
        record = _read(path)
        if not record:
            return None
        matrix = record["clean"]["confusion_matrix"] if dataset == "fashion" else record["confusion_matrix"]
        records.append(np.asarray(matrix))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sum(records), cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="True")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    return fig


def _render_dataset(dataset: str, classes):
    result = _read(ROOT / "results" / "summary" / f"{dataset}_final.json")
    cost = _read(ROOT / "results" / "summary" / f"{dataset}_costs.json")
    if not result:
        st.warning("正式实验结果文件缺失，无法展示指标。")
        return
    summary = result["summary"]
    modes = ["A", "B", "C25", "C"] if dataset == "fashion" else ["A", "B", "C"]
    rows = []
    for mode in modes:
        item = summary[mode]
        prefix = "clean_" if dataset == "fashion" else ""
        rows.append({"组别": mode,
                     "Accuracy 均值": item[f"{prefix}accuracy_mean"],
                     "Accuracy 标准差": item[f"{prefix}accuracy_std"],
                     "Macro-F1 均值": item[f"{prefix}macro_f1_mean"],
                     "Macro-F1 标准差": item[f"{prefix}macro_f1_std"]})
    frame = pd.DataFrame(rows)
    st.dataframe(frame.style.format({key: "{:.4f}" for key in frame.columns if key != "组别"}),
                 width="stretch", hide_index=True)
    st.bar_chart(frame.set_index("组别")[["Accuracy 均值", "Macro-F1 均值"]], stack=False)
    if dataset == "fashion":
        st.markdown("#### 生成比例消融")
        ablation = pd.DataFrame({"生成比例": [0, 25, 50],
                                 "Accuracy": [summary[m]["clean_accuracy_mean"] for m in ("A", "C25", "C")],
                                 "Macro-F1": [summary[m]["clean_macro_f1_mean"] for m in ("A", "C25", "C")]})
        st.line_chart(ablation.set_index("生成比例"))
        st.markdown("#### 固定高斯噪声（σ=0.15）")
        noisy = pd.DataFrame({"组别": modes,
                              "干净 Accuracy": [summary[m]["clean_accuracy_mean"] for m in modes],
                              "加噪 Accuracy": [summary[m]["noisy_accuracy_mean"] for m in modes]})
        st.bar_chart(noisy.set_index("组别"), stack=False)
    st.markdown("#### 各类别 Recall")
    recall_key = "clean_recall_mean" if dataset == "fashion" else "recall_mean"
    recall = pd.DataFrame({mode: summary[mode][recall_key] for mode in modes}, index=classes)
    st.dataframe(recall.style.format("{:.3f}"), width="stretch")
    st.line_chart(recall)
    st.markdown("#### 混淆矩阵（三种子合计）")
    chosen = st.selectbox("选择组别", modes, key=f"{dataset}_confusion_mode")
    fig = _confusion(dataset, chosen, classes)
    if fig:
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("混淆矩阵原始文件缺失。")
    if cost:
        st.markdown("#### 分类器训练时间")
        seconds = pd.DataFrame({"组别": modes,
                                "模型循环耗时（秒）": [cost["cnn_training"][m]["seconds_mean"] for m in modes]})
        st.bar_chart(seconds.set_index("组别"))
        st.caption("时间不含 Python 启动与数据加载；生成器训练与采样成本另计，详情见报告。")


def render():
    st.title("实验结果对比")
    st.caption("所有数值从正式实验 JSON 读取；Fashion 与 NEU 使用独立模型和数据划分。")
    fashion, neu = st.tabs(["Fashion-MNIST 课程实验", "NEU-CLS 工业实验"])
    with fashion:
        _render_dataset("fashion", [str(i) for i in range(10)])
    with neu:
        _render_dataset("neu", ["Cr", "In", "Pa", "PS", "RS", "Sc"])
