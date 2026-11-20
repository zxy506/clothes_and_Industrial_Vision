from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.services.history_service import save_predictions, to_csv_bytes
from src.services.inference_service import CLASSES, checkpoint_path, predict


def _read_image(upload):
    raw = upload.getvalue()
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("文件超过 20 MB")
    with Image.open(BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG", "BMP"}:
            raise ValueError("仅支持 PNG、JPEG、BMP")
        image.verify()
    with Image.open(BytesIO(raw)) as image:
        return image.copy()


def render(load_classifier_cached):
    st.title("工业缺陷识别")
    st.caption("上传单张或多张钢材表面图像；系统将统一转为 128×128 灰度图进行图像级分类。")
    col1, col2 = st.columns(2)
    mode = col1.selectbox("CNN 训练策略", ("A", "B", "C"), format_func=lambda x: {
        "A": "A · 真实数据", "B": "B · 常规增强", "C": "C · CVAE 增强"}[x])
    seed = col2.selectbox("模型种子", (0, 1, 2))
    path = checkpoint_path(mode, seed)
    if not path.exists():
        st.error(f"模型文件缺失：{path.name}。请先训练对应权重。")
    uploads = st.file_uploader("上传图片（可多选）", type=["png", "jpg", "jpeg", "bmp"],
                               accept_multiple_files=True)
    if uploads:
        st.write(f"已选择 {len(uploads)} 张图片")
        if len(uploads) > 50:
            st.warning("单次最多处理前 50 张，请分批上传。")
        previews, preview_names = [], []
        for upload in uploads[:6]:
            try:
                previews.append(_read_image(upload))
                preview_names.append(upload.name)
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
                st.warning(f"{upload.name} 无法预览：{exc}")
        if previews:
            st.image(previews, caption=preview_names, width=160)
    if st.button("开始识别", type="primary", disabled=not uploads or not path.exists()):
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
            predictions = predict(model, device, images)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(f"模型加载或识别失败：{exc}")
            return
        rows = [{"image_name": name, **prediction} for name, prediction in zip(names, predictions)]
        saved = save_predictions(rows, mode=mode, seed=seed, version=version)
        st.session_state["latest_detection"] = {"rows": rows, "names": names,
                                                 "images": images, "mode": mode, "seed": seed,
                                                 "version": version}
        st.success(f"完成 {len(rows)} 张识别，已保存 {saved} 条历史记录。")
    latest = st.session_state.get("latest_detection")
    if not latest or latest["mode"] != mode or latest["seed"] != seed:
        return
    st.markdown("### 本次结果")
    frame = pd.DataFrame([{"图片": row["image_name"], "类别": f"{row['class_zh']} ({row['class_en']})",
                           "预测概率": row["confidence"], "模型": f"{latest['mode']}-seed{latest['seed']}"}
                          for row in latest["rows"]])
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("导出本次结果 CSV", to_csv_bytes(frame), "industrial_vision_results.csv", "text/csv")
    for index, (image, row) in enumerate(zip(latest["images"], latest["rows"])):
        with st.expander(f"{row['image_name']} · {row['class_zh']} · {row['confidence']:.1%}",
                         expanded=index == 0):
            left, right = st.columns([1, 2])
            left.image(image, width=240)
            probabilities = pd.DataFrame({"类别": [f"{c[2]} ({c[0]})" for c in CLASSES],
                                          "概率": [row["probabilities"][c[0]] for c in CLASSES]})
            right.bar_chart(probabilities.set_index("类别"))
            right.caption(f"模型版本：{latest['version']}；仅输出整图类别，无缺陷定位。")
