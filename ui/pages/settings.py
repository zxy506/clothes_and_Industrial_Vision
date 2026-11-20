from pathlib import Path

import streamlit as st
import torch
import yaml

from src.services.inference_service import ROOT


def render():
    st.title("系统设置")
    config = yaml.safe_load((ROOT / "configs/app.yaml").read_text(encoding="utf-8"))
    st.markdown(f"**计算设备：** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    st.markdown("**输入图片尺寸：** 工业 128×128 灰度；衣物 28×28 灰度")
    st.markdown("**分类模型：** NEUCNN 六类；Fashion CNN 十类。两个任务的权重独立。")
    st.markdown("**生成模型：** NEUCVAE，六类。")
    st.markdown(f"**数据库：** {config['app']['database']}")
    st.markdown("### 模型加载状态")
    for mode in "ABC":
        st.write(f"{mode}：" + " · ".join(
            f"seed {seed} {'就绪' if (ROOT / f'checkpoints/neu/cnn_{mode}_seed{seed}.pt').exists() else '缺失'}"
            for seed in (0, 1, 2)))
    generator = ROOT / "checkpoints/neu/cvae_seed0.pt"
    st.write(f"CVAE：{'就绪' if generator.exists() else '缺失'}")
    st.markdown("### 衣物模型状态")
    for mode in ("A", "B", "C25", "C"):
        st.write(f"{mode}：" + " · ".join(
            f"seed {seed} {'就绪' if (ROOT / f'checkpoints/fashion/cnn_{mode}_seed{seed}_30epochs.pt').exists() else '缺失'}"
            for seed in (0, 1, 2)))
    st.markdown("### 运行配置")
    st.code(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), language="yaml")
    st.caption("模型文件缺失时，识别和生成页会显示错误，不会提供虚构结果。")
