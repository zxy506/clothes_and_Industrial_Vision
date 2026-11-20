from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

from src.services.generation_service import generate, save_generated
from src.services.inference_service import CLASSES, ROOT


def render(load_generator_cached):
    st.title("CVAE 样本生成")
    st.caption("使用 NEU-CLS 小样本训练的条件 VAE。生成图可能模糊，仅供研究展示与数据增强实验。")
    path = ROOT / "checkpoints/neu/cvae_seed0.pt"
    if not path.exists():
        st.error(f"生成器文件缺失：{path.name}。请先训练 NEU CVAE。")
    names = {code: f"{zh} · {english} ({code})" for code, english, zh in CLASSES}
    left, middle, right = st.columns(3)
    class_code = left.selectbox("缺陷类别", tuple(names), format_func=lambda code: names[code])
    count = middle.number_input("生成数量", min_value=1, max_value=48, value=6, step=1)
    seed = right.number_input("随机种子", min_value=0, max_value=2**31 - 1, value=0, step=1)
    if st.button("生成样本", type="primary", disabled=not path.exists()):
        try:
            model, device, version = load_generator_cached(path.stat().st_mtime_ns)
            images = generate(model, device, class_code, int(count), int(seed))
            folder, metadata = save_generated(images, class_code, int(seed), version)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            st.error(f"生成失败：{exc}")
            return
        archive = BytesIO()
        with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
            for index, image in enumerate(images):
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                bundle.writestr(metadata["files"][index], buffer.getvalue())
            bundle.writestr("metadata.json", (folder / "metadata.json").read_bytes())
        st.session_state["latest_generation"] = {"images": images, "archive": archive.getvalue(),
                                                  "folder": str(folder), "class_code": class_code,
                                                  "seed": int(seed), "version": version}
    latest = st.session_state.get("latest_generation")
    if latest:
        st.success(f"已生成 {len(latest['images'])} 张，保存在 {latest['folder']}")
        st.image(latest["images"], caption=[f"{latest['class_code']} #{i + 1}" for i in range(len(latest["images"]))],
                 width=160)
        st.download_button("下载生成图片 ZIP", latest["archive"],
                           f"NEU_{latest['class_code']}_seed{latest['seed']}.zip", "application/zip")
        st.caption(f"模型版本：{latest['version']} · 种子：{latest['seed']}")
