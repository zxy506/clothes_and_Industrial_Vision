"""Fashion-MNIST ten-class recognition page."""

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.services.fashion_inference_service import CLASSES, checkpoint_path, predict
from src.services.history_service import save_predictions, to_csv_bytes
from ui.pages.recognition import _read_image


def render(load_classifier_cached):
    st.title("衣物识别")
    st.caption("使用独立训练的 Fashion-MNIST 十分类 CNN。适合 28×28、灰度、黑色背景上的单件衣物；普通照片可能与训练数据差异很大。")
    left, right = st.columns(2)
    mode = left.selectbox("衣物 CNN 训练策略", ("A", "B", "C25", "C"), format_func=lambda x: {
        "A": "A · 真实数据", "B": "B · 常规增强", "C25": "C25 · 25% CVAE 增强",
        "C": "C50 · 50% CVAE 增强"}[x])
    seed = right.selectbox("衣物模型种子", (0, 1, 2))
    path = checkpoint_path(mode, seed)
    if not path.is_file():
        st.error(f"衣物模型文件缺失：{path.name}。请先训练对应权重。")
    uploads = st.file_uploader("上传衣物图片（可多选）", type=["png", "jpg", "jpeg", "bmp"],
                               accept_multiple_files=True, key="fashion_uploads")
    if uploads:
        st.write(f"已选择 {len(uploads)} 张图片")
        if len(uploads) > 50:
            st.warning("单次最多处理前 50 张，请分批上传。")
        previews, names = [], []
        for upload in uploads[:10]:
            try:
                previews.append(_read_image(upload))
                names.append(upload.name)
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
                st.warning(f"{upload.name} 无法预览：{exc}")
        if previews:
            st.image(previews, caption=names, width=110)
    if st.button("开始衣物识别", type="primary", disabled=not uploads or not path.is_file()):
        images, names = [], []
        for upload in uploads[:50]:
            try:
                images.append(_read_image(upload))
                names.append(upload.name)
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
                st.error(f"{upload.name} 无法处理：{exc}")
        if not images:
            st.error("没有可识别的图片。")
            return
        try:
            model, device, version = load_classifier_cached(mode, seed, path.stat().st_mtime_ns)
            rows = [{"image_name": name, **result} for name, result in
                    zip(names, predict(model, device, images))]
            saved = save_predictions(rows, mode=mode, seed=seed, version=version,
                                     model_family="FashionCNN")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(f"衣物模型加载或识别失败：{exc}")
            return
        st.session_state["latest_fashion"] = {"rows": rows, "images": images,
                                              "mode": mode, "seed": seed, "version": version}
        st.success(f"完成 {len(rows)} 张衣物识别，已保存 {saved} 条历史记录。")
    latest = st.session_state.get("latest_fashion")
    if not latest or latest["mode"] != mode or latest["seed"] != seed:
        return
    st.markdown("### 本次结果")
    frame = pd.DataFrame([{"图片": row["image_name"],
                           "类别": f"{row['class_zh']} ({row['class_en']})",
                           "预测概率": row["confidence"], "模型": f"{mode}-seed{seed}"}
                          for row in latest["rows"]])
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("导出本次衣物结果 CSV", to_csv_bytes(frame),
                       "fashion_recognition_results.csv", "text/csv")
    for index, (image, row) in enumerate(zip(latest["images"], latest["rows"])):
        with st.expander(f"{row['image_name']} · {row['class_zh']} · {row['confidence']:.1%}",
                         expanded=index == 0):
            left, right = st.columns([1, 2])
            left.image(image, width=224)
            probabilities = pd.DataFrame({"类别": [f"{item[2]} ({item[1]})" for item in CLASSES],
                                          "概率": [row["probabilities"][item[0]] for item in CLASSES]})
            right.bar_chart(probabilities.set_index("类别"))
            right.caption(f"模型版本：{latest['version']}；输入统一缩放为 28×28 灰度图。")
