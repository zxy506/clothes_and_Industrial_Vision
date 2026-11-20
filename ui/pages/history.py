import streamlit as st

from src.services.history_service import query_history, to_csv_bytes
from src.services.inference_service import CLASSES
from src.services.fashion_inference_service import CLASSES as FASHION_CLASSES


def render():
    st.title("历史检测记录")
    st.caption("记录保存在本项目的 SQLite 数据库中，时间采用 Asia/Shanghai。")
    dataset = st.selectbox("任务", ("工业缺陷", "衣物识别"))
    family = "NEUCNN" if dataset == "工业缺陷" else "FashionCNN"
    classes = CLASSES if dataset == "工业缺陷" else FASHION_CLASSES
    options = {"全部": None, **{f"{zh} ({code})": code for code, _, zh in classes}}
    left, right = st.columns(2)
    selected = left.selectbox("按类别筛选", list(options))
    limit = right.selectbox("最多显示", (100, 500, 1000, 5000))
    try:
        frame = query_history(class_code=options[selected], limit=limit, model_family=family)
    except Exception as exc:
        st.error(f"历史记录读取失败：{exc}")
        return
    if frame.empty:
        st.info("还没有符合条件的检测记录。")
        return
    visible = frame.rename(columns={"detected_at": "检测时间", "image_name": "图片名称",
                                    "class_code": "类别代码", "class_zh": "预测类别",
                                    "confidence": "预测概率", "model_name": "模型名称",
                                    "model_version": "模型版本"})
    st.dataframe(visible[["检测时间", "图片名称", "预测类别", "类别代码", "预测概率", "模型名称", "模型版本"]],
                 width="stretch", hide_index=True)
    st.download_button("导出当前查询 CSV", to_csv_bytes(visible),
                       "industrial_vision_history.csv", "text/csv")
